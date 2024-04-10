# forked from https://github.com/txie-93/cgcnn
from __future__ import print_function, division
import torch
import torch.nn as nn
import argparse
import os
import sys
import pathlib
try:
    from torch_scatter import scatter
except BaseException:
    from torch_geometric.utils import scatter

from torch_geometric.utils import softmax

roots = pathlib.Path(__file__).parent.parent
sys.path.append(roots)  # append top directory


class ConvLayer(torch.nn.Module):
    """
    Convolutional operation on graphs
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len,
                                 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.act1 = nn.Softplus()
        # ; PairNorm; GraphNorm; DiffGroupNorm
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.act2 = nn.Softplus()
        self.act3 = softmax

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx, batch):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len) -->(nodes, atom_fea_len) a.k.a. x
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len) --> (edges, nbr_fea_len) a.k.a. edge_attr
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M) --> (2, edges) a.k.a. edge_index
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        row, col = nbr_fea_idx
        # convolution
        atom_self_fea = atom_in_fea[row]  # (edges,atom_fea_len)
        atom_nbr_fea = atom_in_fea[col]  # (edges,atom_fea_len)
        total_nbr_fea = torch.cat(
            (atom_self_fea, atom_nbr_fea, nbr_fea), dim=-1)  # (edges, DIM)
        total_gated_fea = self.fc_full(total_nbr_fea)  # (edges, DIM)
        # (edges, self.atom_fea_len*2)
        total_gated_fea = self.bn1(total_gated_fea)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=-1)
        # print(nbr_filter.shape, row.shape)
        # nbr_filter = self.act3(nbr_filter.sum(dim=-1), row)[:,None] *
        # nbr_filter ####NEW: attention
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.act1(nbr_core)
        # nbr_core = self.act3(nbr_core.sum(dim=-1), row)[:,None] * nbr_core
        # #####NEW: attention
        nbr_summed = scatter(
            src=nbr_filter * nbr_core,
            index=col,
            dim=0)  # (nodes, dim)
        nbr_summed = self.bn2(nbr_summed)
        out = self.act2(atom_in_fea + nbr_summed)  # (nodes, dim)
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """

    def __init__(self, orig_atom_fea_len=92, nbr_fea_len=41,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False, explain=False, mean=None, std=None, learnable=False, **kwargs):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.learnable = learnable
        self.embedding = nn.Linear(
            orig_atom_fea_len,
            atom_fea_len) if not learnable else torch.nn.Embedding(
            120,
            atom_fea_len)
        # self.embedding = torch.nn.Embedding(120, atom_fea_len)
        self.explain = explain

        if self.explain:
            def hook(module, inputs, grad):
                self.embedded_grad = grad
            self.embedding.register_backward_hook(hook)

        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                            for _ in range(n_h - 1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        self.mean = mean
        self.std = std

        self.reset_all_weights()
        self.return_property = kwargs.get("return_property", True)

    def reset_all_weights(self, ) -> None:
        """
        refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        """

        @torch.no_grad()
        def weight_reset(m: torch.nn.Module):
            # - check if the current module has reset_parameters & if it's callabed called it on m
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        # Applies fn recursively to every submodule see:
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        self.apply(fn=weight_reset)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx,
                dists, crystal_atom_idx, batch=None):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len) --> (nodes, dim)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)  --> (edges, dim)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M) --> (2,edges)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx --> useless
        batch: torch.LongTensor

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        atom_fea = self.embedding(atom_fea) if not self.learnable else self.embedding(
            crystal_atom_idx)  # from dictionary of dataloader!
        # atom_fea = self.embedding(crystal_atom_idx)
        for conv_func in self.convs:
            atom_fea = conv_func(
                atom_fea,
                nbr_fea,
                nbr_fea_idx,
                batch)  # nodes, dim

        if self.explain:
            self.final_conv_acts = atom_fea

            def hook(grad):
                self.final_conv_grads = grad
            self.final_conv_acts.register_hook(hook)  # only when backpropped!

        if not self.return_property:
            return atom_fea  # nodes, dim

        crys_fea = self.pooling(atom_fea, batch)  # nmolecues, dim

        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))

        out = self.fc_out(crys_fea)  # B,1

        if self.mean is not None and self.std is not None:
            out = out * self.std + self.mean

        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
#         assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
#             atom_fea.data.shape[0]
#         summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
#                       for idx_map in crystal_atom_idx]
#         return torch.cat(summed_fea, dim=0)
        return scatter(src=atom_fea, index=crystal_atom_idx,
                       dim=0, reduce="mean")  # (nmolecules, dim)


if __name__ == "__main__":
    def get_parser():
        parser = argparse.ArgumentParser()
        #     parser.add_argument('--name', type=str, default=''.join(random.choice(string.ascii_lowercase) for i in range(10)))
        parser.add_argument('--name', type=str, default=None)
        parser.add_argument('--seed', type=int, default=7)
        parser.add_argument('--gpu', action='store_true')
        parser.add_argument('--gpus', action='store_true')
        parser.add_argument('--silent', action='store_true')
        # only returns true when passed in bash
        parser.add_argument('--log', action='store_true')
        parser.add_argument('--plot', action='store_true')
        parser.add_argument(
            '--use-artifacts',
            action='store_true',
            help="download model artifacts for loading a model...")

        # data
        parser.add_argument('--train_test_ratio', type=float, default=0.02)
        parser.add_argument('--train_val_ratio', type=float, default=0.03)
        parser.add_argument('--train_frac', type=float, default=0.8)
        parser.add_argument('--warm_up_split', type=int, default=5)
        parser.add_argument('--batches', type=int, default=160)
        parser.add_argument(
            '--test_samples',
            type=int,
            default=5)  # -1 for all
        parser.add_argument('--test_steps', type=int, default=100)
        parser.add_argument(
            '--data_norm',
            action='store_true')  # normalize energy???
        parser.add_argument(
            '--dataset',
            type=str,
            default="qm9edge",
            choices=[
                "qm9",
                "md17",
                "ani1",
                "ani1x",
                "qm9edge",
                "moleculenet"])
        parser.add_argument(
            '--data_dir',
            type=str,
            default="/Scr/hyunpark/ArgonneGNN/argonne_gnn/data")
        parser.add_argument('--task', type=str, default="homo")
        # causes CUDAMemory error;; asynchronously reported at some other API
        # call
        parser.add_argument('--pin_memory', type=bool, default=True)
        parser.add_argument(
            '--use_artifacts',
            action="store_true",
            help="use artifacts for resuming to train")
        # for data, use DGL or PyG formats?
        parser.add_argument('--use_tensors', action="store_true")

        # train
        parser.add_argument('--epoches', type=int, default=2)
        parser.add_argument(
            '--batch_size',
            type=int,
            default=128)  # Per GPU batch size
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
        parser.add_argument('--weight_decay', type=float, default=2e-5)
        parser.add_argument('--dropout', type=float, default=0)
        parser.add_argument('--resume', action='store_true')
        parser.add_argument('--distributed', action="store_true")
        parser.add_argument('--low_memory', action="store_true")
        parser.add_argument(
            '--amp',
            action="store_true",
            help="floating 16 when turned on.")
        parser.add_argument(
            '--loss_schedule',
            '-ls',
            type=str,
            choices=[
                "manual",
                "lrannealing",
                "softadapt",
                "relobralo",
                "gradnorm"],
            help="how to adjust loss weights.")
        parser.add_argument('--with_force', type=bool, default=False)
        parser.add_argument(
            '--optimizer',
            type=str,
            default='adam',
            choices=[
                "adam",
                "lamb",
                "sgd",
                "torch_adam"])
        parser.add_argument('--gradient_clip', type=float, default=None)
        parser.add_argument('--accumulate_grad_batches', type=int, default=1)
        parser.add_argument(
            '--shard',
            action="store_true",
            help="fairscale ShardedDDP")  # fairscale ShardedDDP?
        parser.add_argument(
            "--not_use_env",
            default=False,
            action="store_false",
            help="Use environment variable to pass "
            "'local rank'. For legacy reasons, the default value is False. "
            "If set to True, the script will not pass "
            "--local_rank as argument, and will instead set LOCAL_RANK.",
        )

        # model
        parser.add_argument(
            '--backbone',
            type=str,
            default='physnet',
            choices=[
                "schnet",
                "physnet",
                "torchmdnet",
                "alignn",
                "dimenet",
                "dimenetpp"])
        parser.add_argument(
            '--load_ckpt_path',
            type=str,
            default="/Scr/hyunpark/ArgonneGNN/argonne_gnn/save/")
        parser.add_argument(
            '--explain',
            type=bool,
            default=False,
            help="gradient hook for CAM...")  # Only for Schnet.Physnet.Alignn WIP!

        # hyperparameter optim
        parser.add_argument(
            '--resume_hp',
            action="store_true",
            help="resume hp search if discontinued...")
        parser.add_argument(
            '--hp_savefile',
            type=str,
            default="results.csv",
            help="resume hp search from this file...")
        opt = parser.parse_args()
        return opt

    opt = get_parser()

    config = dict(orig_atom_fea_len=92, nbr_fea_len=41,
                  atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                  classification=False)
    m = CrystalGraphConvNet(**config)

    from train.general_train_eval import load_state
    path_and_name = os.path.join("./", "{}.pth".format("cgcnn"))
    load_state(m, None, None, path_and_name, model_only=True)
    m.eval()

    # Get Dataloader
    import train.data_utils as cdu
    root_dir = "../cif_files"
    dataset = cdu.CIFData(root_dir)
    dataloader = cdu.get_dataloader(dataset, shuffle=False, **{'pin_memory': opt.pin_memory, 'persistent_workers': False,
                                                               'batch_size': opt.batch_size})
    li = iter(dataloader).next()

    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, batch, dists, y = li.x, li.edge_attr, li.edge_index, li.cif_id, li.batch, li.edge_weight, li.y
    result = m(atom_fea, nbr_fea, nbr_fea_idx, dists, crystal_atom_idx, batch)
    print(result.view(-1), y)
    print(result.view(-1) - y)
