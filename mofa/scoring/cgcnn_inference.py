import os
import ase
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from train.dist_utils import get_local_rank, init_distributed
import torch.nn as nn
import torch.distributed as dist
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from typing import Union
from pathlib import Path
from models import cgcnn, CrystalGraphConvNet
import os
import abc
import json
import torch
import pickle
import warnings
import pandas as pd
import numpy as np

_atom_init_dir = Path(__file__).parent / "files"
_cgcnn_models_dir = (Path(__file__).parent / ".." / ".." / "models" / "cgcnn-hmof-0.1bar-300k").resolve()


class Opt:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def _get_split_sizes(
        train_frac: float, full_dataset: Dataset) -> tuple[int, int, int]:
    """DONE: Need to change split schemes!"""
    len_full = len(full_dataset)
    len_train = int(len_full * train_frac)
    len_test = int(0.1 * len_full)
    len_val = len_full - len_train - len_test
    return len_train, len_val, len_test


def get_dataloader(dataset: Dataset, shuffle: bool,
                   collate_fn: callable = None, **kwargs):
    sampler = DistributedSampler(
        dataset, shuffle=shuffle) if dist.is_initialized() else None
    loader = DataLoader(
        dataset,
        shuffle=(
            shuffle and sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
        **kwargs)
    return loader


class GaussianDistance(torch.nn.Module):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
#         self.filter = np.arange(dmin, dmax+step, step)
        super().__init__()
        self.register_buffer(
            "filter",
            torch.arange(
                dmin,
                dmax + step,
                step).type(
                torch.float32))
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return torch.exp(-(distances[..., None] - self.filter)**2 /
                         self.var**2)

    def forward(self, distances):
        return self.expand(distances)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = torch.from_numpy(
                np.array(value)).type(torch.float32)


class CIFData(Dataset):
    """
    Torch Geometric Data is return
    """

    def __init__(self, list_ase_atoms,
                 cifnames,
                 target=None,
                 root_dir=os.path.join(os.getcwd(), "cif_files"),
                 max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123):
        super().__init__()
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.list_ase_atoms = list_ase_atoms
        self.cifnames = cifnames
        if isinstance(target, type(None)):
            self.target = [0. for x in range(0, len(self.list_ase_atoms))]
        else:
            self.target = target
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)  # Embedding dict!
        self.gdf = GaussianDistance(
            dmin=dmin,
            dmax=self.radius,
            step=step)  # nn.Module

    def len(self):
        return len(self.list_ase_atoms)

    def get(self, idx):
        ase_atoms = self.list_ase_atoms[idx]
        cif_id = self.cifnames[idx]
        target = self.target[idx]
        try:
            if ".cif" in cif_id:
                cif_id = os.path.splitext(cif_id)[0]
            crystal = AseAtomsAdaptor.get_structure(ase_atoms)
            atom_idx = [
                crystal.sites[i].specie.number for i in range(
                    len(crystal))]
            atom_fea = np.vstack([self.ari.get_atom_fea(crystal.sites[i].specie.number)
                                  for i in range(len(crystal))])  # Embed atoms of crystal
            atom_fea = torch.from_numpy(atom_fea)  # tensorfy
            # list has num_atoms elements; each element has variable length
            # neighbors!
            all_nbrs = crystal.get_all_neighbors(
                self.radius, include_index=True)
            # For each atom of crystal, sort each atomic neighbors based on
            # distance!
            all_nbrs = [sorted(nbrs, key=lambda x: x[1])
                        for nbrs in all_nbrs]
            nbr_fea_idx_row, nbr_fea_idx_col, nbr_fea = [], [], []

            for idx, nbr in enumerate(all_nbrs):
                if len(nbr) < self.max_num_nbr:
                    warnings.warn('{} not find enough neighbors to build graph. '
                                  'If it happens frequently, consider increase '
                                  'radius.'.format(cif_id))
                nbr_fea_idx_row.extend([idx] * len(nbr))  # num_edges
                nbr_fea_idx_col.extend(
                    list(map(lambda x: x[2], nbr)))  # num_edges
                nbr_fea.extend(list(map(lambda x: x[1], nbr)))  # num_edges

            nbr_fea_idx_row, nbr_fea_idx_col, nbr_fea = np.array(nbr_fea_idx_row), np.array(nbr_fea_idx_col), torch.from_numpy(
                np.array(nbr_fea))  # (n_i, M), (n_i, atom_fea_len) --> (edges=n_i*M,), (edges=n_i*M,), (edges=n_i*M, atom_fea_len)
            dists = nbr_fea.type(torch.float32)  # edges,
            # (n_i, M, nbr_fea_len) --> (edges, nbr_fea_len)
            nbr_fea = self.gdf.expand(nbr_fea)
            atom_fea = torch.tensor(atom_fea).type(
                torch.float32)  # (natoms, atom_fea_len)
            nbr_fea = torch.tensor(nbr_fea).type(
                torch.float32)  # (edges, nbr_fea_len)
            nbr_fea_idx_row = torch.LongTensor(
                nbr_fea_idx_row).type(torch.long)  # edges,
            nbr_fea_idx_col = torch.LongTensor(
                nbr_fea_idx_col).type(torch.long)  # edges,
            nbr_fea_idx = torch.stack(
                (nbr_fea_idx_row, nbr_fea_idx_col), dim=0)  # (2,edges)
            target = torch.Tensor([float(target)]).type(
                torch.float32)  # (1,) so that (B,1) when batched!
            atom_idx = torch.from_numpy(
                np.array(atom_idx)).type(
                torch.long)  # nodes,
            return Data(x=atom_fea, edge_attr=nbr_fea, edge_index=nbr_fea_idx,
                        edge_weight=dists, y=target, cif_id=atom_idx, ), cif_id  # PyG type dataset
        except Exception as e:
            print(e)


class DataModuleCrystal(abc.ABC):
    """ Abstract DataModule. Children must define self.ds_{train | val | test}. """

    def __init__(self, ase_mofs, mof_names, **dataloader_kwargs):
        super().__init__()
        self.opt = opt = dataloader_kwargs.pop("opt")
        self.ase_mofs = ase_mofs
        self.mof_names = mof_names
        if get_local_rank() == 0:
            self.prepare_data()
            print(f"{get_local_rank()}-th core is parsed!")

        # Wait until rank zero has prepared the data (download, preprocessing,
        # ...)
        if dist.is_initialized():
            # WAITNG for 0-th core is done!
            dist.barrier(device_ids=[get_local_rank()])

        # change to data_dir and DEPRECATE this command
        root_dir = self.opt.data_dir_crystal
        print("Root dir chosen is", root_dir)
        if self.opt.dataset in ["cifdata"]:
            if self.opt.save_to_pickle is None:
                full_dataset = CIFData(
                    self.ase_mofs, self.mof_names, root_dir=root_dir)
            else:
                pickle_data = os.path.splitext(self.opt.save_to_pickle)[0]
                pickle_data += ".pickle"
                if not os.path.exists(pickle_data):
                    print("Saving a pickle file!")
                    full_dataset = CIFData(
                        self.ase_mofs, self.mof_names, root_dir=root_dir)
                    with open(pickle_data, "wb") as f:
                        pickle.dump(full_dataset, f)
                else:
                    print("Loading a saved pickle file!")

                    class CustomUnpickler(pickle.Unpickler):
                        def find_class(self, module, name):
                            if name == 'CIFData':
                                return CIFData
                            return super().find_class(module, name)
                    full_dataset = CustomUnpickler(
                        open(pickle_data, "rb")).load()

        self.dataloader_kwargs = {
            'pin_memory': opt.pin_memory,
            'persistent_workers': dataloader_kwargs.get(
                'num_workers',
                0) > 0,
            'batch_size': opt.batch_size}
        if self.opt.dataset not in ["cifdata"]:
            self.ds_train, self.ds_val, self.ds_test = torch.utils.data.random_split(full_dataset, _get_split_sizes(self.opt.train_frac, full_dataset),
                                                                                     generator=torch.Generator().manual_seed(0))
        else:

            self.ds_train, self.ds_val, self.ds_test = torch.utils.data.random_split(full_dataset, _get_split_sizes(self.opt.train_frac, full_dataset),
                                                                                         generator=torch.Generator().manual_seed(0))

        self._mean = None
        self._std = None

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def prepare_data(self, ):
        """ Method called only once per node. Put here any downloading or preprocessing """
        root_dir = self.opt.data_dir_crystal
        if self.opt.dataset in ["cifdata"]:
            CIFData(self.ase_mofs, self.mof_names, root_dir=root_dir)

    def train_dataloader(self) -> DataLoader:
        if self.opt.data_norm:
            print("Not applicable for crystal...")
        return get_dataloader(self.ds_train, shuffle=False,
                              collate_fn=None, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        if self.opt.data_norm:
            print("Not applicable for crystal...")
        return get_dataloader(self.ds_val, shuffle=False,
                              collate_fn=None, **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        if self.opt.data_norm:
            print("Not applicable for crystal...")
        return get_dataloader(self.ds_test, shuffle=False,
                              collate_fn=None, **self.dataloader_kwargs)


def infer_for_crystal(opt, dataloader, model):
    device = torch.device("cpu")
    if opt.gpu:
        device = torch.cuda.current_device()
    df_list = []
    for one_data_batch in dataloader:
        data_batch = one_data_batch[0]  # Get DATA instance
        data_names = one_data_batch[1]  # Get CIF names
        data_batch = data_batch.to(device)

        if opt.ensemble_names is not None:
            e, s = model(data_batch.x, data_batch.edge_attr, data_batch.edge_index,
                         data_batch.edge_weight, data_batch.cif_id, data_batch.batch)
            energies = e
            stds = s
        else:
            e = model(
                data_batch.x,
                data_batch.edge_attr,
                data_batch.edge_index,
                data_batch.edge_weight,
                data_batch.cif_id,
                data_batch.batch)
            energies = e
        y = data_batch.y

        if opt.ensemble_names is not None:
            df_list = df_list + [pd.DataFrame(data=np.concatenate([np.array(data_names).reshape(-1, 1), energies.detach().cpu().numpy().reshape(-1, 1),
                                                                   stds.detach().cpu().numpy().reshape(-1, 1), y.detach().cpu().numpy().reshape(-1, 1)], axis=1),
                                              columns=["name", "pred", "std", "real"])]
        else:
            df_list = df_list + [pd.DataFrame(data=np.concatenate([np.array(data_names).reshape(-1, 1), energies.detach().cpu().numpy().reshape(-1, 1),
                                                                   y.detach().cpu().numpy().reshape(-1, 1)], axis=1), columns=["name", "pred", "real"])]

    df = pd.concat(df_list, axis=0, ignore_index=True)

    df["name"] = df["name"].astype(int)
    return df.sort_values(by="name")


def load_state(model: nn.Module,
               path_and_name: Union[Path, str], model_only=False):
    ckpt = torch.load(
        path_and_name, map_location={
            'cuda:0': f'cuda:{get_local_rank()}'})
    try:
        if isinstance(model, DistributedDataParallel):
            model.module.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt["model"])
        epoch = 0
        val_loss = 1e20
    except Exception as e:
        print(e)
        if isinstance(model, DistributedDataParallel):
            model.module.load_state_dict(ckpt)
        else:
            model.load_state_dict(ckpt)
        epoch = 0
        val_loss = 1e20
    finally:
        print(f"Loaded a model from rank {get_local_rank()}!")
    return epoch, val_loss


def call_loader(ase_mofs, mof_names, opt: Opt):
    datamodule = DataModuleCrystal(
        ase_mofs=ase_mofs, mof_names=mof_names, opt=opt)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    mean = datamodule.mean
    std = datamodule.std
    return train_loader, val_loader, test_loader, mean, std


def call_model(opt: Opt, mean: float,
               std: float, return_metadata=False):
    # Model
    model = BACKBONES.get(opt.backbone, CrystalGraphConvNet)
    model_kwargs = BACKBONE_KWARGS.get(opt.backbone, None)
    model_kwargs.update({"explain": False})
    model_kwargs.update({"mean": mean, "std": std})
    model = model(**model_kwargs)
    model_kwargs.get("cutoff", 10.)
    model_kwargs.get("max_num_neighbors", 32)
    device = torch.device("cpu")
    if opt.gpu:
        device = torch.cuda.current_device()
        model.to(device)
    model.eval()
    path_and_name = os.path.join(opt.load_ckpt_path, "{}.pth".format(opt.name))
    load_state(
        model,
        path_and_name=path_and_name,
        model_only=True)
    if torch.__version__.startswith('2.0'):
        model = torch.compile(model)
        print("PyTorch model has been compiled...")
    return model


def infer(ase_mofs, mof_names, opt=None):
    init_distributed()
    get_local_rank()
    train_loader, val_loader, test_loader, mean, std = call_loader(
        ase_mofs, mof_names, opt)
    print("mean", mean, "std", std)
    models = []
    for name in opt.ensemble_names:
        opt.name = name
        model = call_model(opt, mean, std)
        models.append(model)
    model = lambda *inp: (torch.cat([models[0](*inp), models[1](*inp), models[2](*inp)], dim=-1).mean(dim=-1),
                          torch.cat([models[0](*inp), models[1](*inp), models[2](*inp)], dim=-1).std(dim=-1))
    if opt.dataset in ["cifdata"]:
        df = infer_for_crystal(opt, train_loader, model)
    return df


BACKBONES = {
    "cgcnn": cgcnn.CrystalGraphConvNet
}

BACKBONE_KWARGS = {
    "cgcnn": dict(orig_atom_fea_len=92, nbr_fea_len=41,
                  atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                  classification=False, learnable=False, explain=False)
}


def run_cgcnn_pred(mofs: list[ase.Atoms], names: list[str], opt: Opt):
    df = infer(mofs, names, opt)
    return df

# manual_batch_size=128 is tested on A40 GPU


def run_cgcnn_pred_wrapper_serial(mofs: list[ase.Atoms], run_name="some_random_string",
                                  manual_batch_size: int = 64, ncpus_to_load_data: int = 1) -> list[float]:
    if len(mofs) == 0:
        return []
    # put mofs into manual batches
    mofnames = [f"{x}" for x in range(0, len(mofs))]
    Nmbatches = int(float(len(mofs)) / manual_batch_size)
    if Nmbatches % manual_batch_size > 0:
        Nmbatches = Nmbatches + 1
    mof_mbatches = [mofs[manual_batch_size *
                         mbatch_i:manual_batch_size *
                         (mbatch_i +
                          1)] for mbatch_i in range(0, Nmbatches -
                                                    1)]
    mof_last_mbatch = mofs[manual_batch_size * (Nmbatches - 1):]
    name_mbatches = [mofnames[manual_batch_size *
                              mbatch_i:manual_batch_size *
                              (mbatch_i +
                               1)] for mbatch_i in range(0, Nmbatches -
                                                         1)]
    name_last_mbatch = mofnames[manual_batch_size * (Nmbatches - 1):]

    if len(mof_last_mbatch) > 0:
        mof_mbatches = mof_mbatches + [mof_last_mbatch]
    if len(name_last_mbatch) > 0:
        name_mbatches = name_mbatches + [name_last_mbatch]
    opt = {
        "name": run_name,
        "ensemble_names": ['cgcnn_pub_hmof_0.1', 'cgcnn_pub_hmof_0.1_dgx', 'cgcnn_pub_hmof_0.1_v2'],
        "gpu": True,
        "data_norm": False,
        "dataset": 'cifdata',
        "train_frac": 1,
        "data_dir_crystal": _atom_init_dir,
        "pin_memory": False,
        "save_to_pickle": None,
        "batch_size": manual_batch_size,
        "num_workers": ncpus_to_load_data,
        "backbone": 'cgcnn',
        "load_ckpt_path": _cgcnn_models_dir,
        "dropnan": False,
    }
    opt = Opt(**opt)
    results = []
    for mbatch_i in range(0, Nmbatches):
        mof_batch = mof_mbatches[mbatch_i]
        name_batch = name_mbatches[mbatch_i]
        if len(mof_batch) == 0:
            continue
        res = run_cgcnn_pred(mof_batch, name_batch, opt=opt)
        results.append(res)
    df = pd.concat(results, axis=0).sort_values(by="name")
    return df["pred"].to_list(), df["std"].to_list()
