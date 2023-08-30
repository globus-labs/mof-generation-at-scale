import argparse
import os
import numpy as np

import torch
import subprocess
from rdkit import Chem

from src import const
from src.datasets import collate_with_fragment_edges, get_dataloader, parse_molecule
from src.lightning import DDPM
# from src.difflinker_lightning import DDPM
from src.linker_size_lightning import SizeClassifier
from src.visualizer import save_xyz_file, visualize_chain
from src.difflinker_molecule_builder import build_molecules
from src.utils import FoundNaNException, split_features
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    '--fragments', action='store', type=str, required=True,
    help='Path to the file with input fragments'
)
parser.add_argument(
    '--model', action='store', type=str, required=True,
    help='Path to the DiffLinker model'
)
parser.add_argument(
    '--linker_size', action='store', type=str, required=True,
    help='Either linker size (int) or path to the GNN for size prediction'
)
parser.add_argument(
    '--output', action='store', type=str, required=False, default='./',
    help='Directory where sampled molecules will be saved'
)
parser.add_argument(
    '--n_samples', action='store', type=int, required=False, default=5,
    help='Number of linkers to generate'
)
parser.add_argument(
    '--n_steps', action='store', type=int, required=False, default=None,
    help='Number of denoising steps'
)
parser.add_argument(
    '--anchors', action='store', type=str, required=False, default=None,
    help='Comma-separated indices of anchor atoms '
         '(according to the order of atoms in the input fragments file, enumeration starts with 1)'
)
parser.add_argument(
    '--samples_dir', action='store', type=str, required=False, default='./',
    help='Directory where sampled molecules ANIMATION will be saved'
)
parser.add_argument(
    '--property_val', action='store', type=float, required=False, default=1.,
    help='WC01 property'
)

def read_molecule(path):
    if path.endswith('.pdb'):
        return Chem.MolFromPDBFile(path, sanitize=False, removeHs=True)
    elif path.endswith('.mol'):
        return Chem.MolFromMolFile(path, sanitize=False, removeHs=True)
    elif path.endswith('.mol2'):
        return Chem.MolFromMol2File(path, sanitize=False, removeHs=True)
    elif path.endswith('.sdf'):
        return Chem.SDMolSupplier(path, sanitize=False, removeHs=True)[0]
    raise Exception('Unknown file extension')


def generate_animation(ddpm, chain_batch, node_mask):
    """e.g. 
    Inside a directory
        ddpm.samples_dir/mol_1/
            mol_1_0.xyz
            mol_1_1.xyz
            mol_1_2.xyz
            mol_1_0.png
            mol_1_1.png
            mol_1_2.png
            mol_1_2.png
            output.gif
    """
    # batch_indices, mol_indices = utils.get_batch_idx_for_animation(self.batch_size, batch_i)

    batch_size = chain_batch.size(1) #Batch size
    batch_indices = torch.arange(batch_size)
    mol_indices = batch_indices
    for bi, mi in zip(batch_indices, mol_indices):
        chain = chain_batch[:, bi, :, :] #(FLD)
        name = f'mol_{mi}'
        chain_output = os.path.join(ddpm.samples_dir, name)
        os.makedirs(chain_output, exist_ok=True)

        one_hot = chain[:, :, 3:-1] if ddpm.include_charges else chain[:, :, 3:] #FLD
        positions = chain[:, :, :3] #FL3
        chain_node_mask = torch.cat([node_mask[bi].unsqueeze(0) for _ in range(ddpm.FRAMES)], dim=0) #FL
        names = [f'{name}_{j}' for j in range(ddpm.FRAMES)]

        save_xyz_file(chain_output, one_hot, positions, chain_node_mask, names=names, is_geom=ddpm.is_geom)
        visualize_chain(chain_output, wandb=None, mode=name, is_geom=ddpm.is_geom) #set wandb None for now!

def main(input_path, model, output_dir, n_samples, n_steps, linker_size, anchors, samples_dir, property_val):

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    if linker_size.isdigit():
        print(f'Will generate linkers with {linker_size} atoms')
        linker_size = int(linker_size)

        def sample_fn(_data):
            return torch.ones(_data['positions'].shape[0], device=device, dtype=const.TORCH_INT) * linker_size
    else:
        print(f'Will generate linkers with sampled numbers of atoms')
        size_nn = SizeClassifier.load_from_checkpoint(linker_size, map_location=device).eval().to(device)

        def sample_fn(_data):
            #same as https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/fce07d701a2d2340f3522df588832c2c0f7e044a/qm9/models.py#L101
            out, _ = size_nn.forward(_data, return_loss=False)
            probabilities = torch.softmax(out, dim=1)
            distribution = torch.distributions.Categorical(probs=probabilities)
            samples = distribution.sample()
            sizes = []
            for label in samples.detach().cpu().numpy():
                sizes.append(size_nn.linker_id2size[label])
            sizes = torch.tensor(sizes, device=samples.device, dtype=const.TORCH_INT)
            return sizes

    ddpm = DDPM.load_from_checkpoint(model, map_location=device, strict=False).eval().to(device)
    ddpm.samples_dir = samples_dir

    if n_steps is not None:
        ddpm.edm.T = n_steps #otherwise, ddpm.edm.T = 1000 default

    if ddpm.center_of_mass == 'anchors' and anchors is None:
        print(
            'Please pass anchor atoms indices '
            'or use another DiffLinker model that does not require information about anchors'
        )
        return

    # Reading input fragments
    extension = input_path.split('.')[-1]
    if extension not in ['sdf', 'pdb', 'mol', 'mol2']:
        print('Please upload the file in one of the following formats: .pdb, .sdf, .mol, .mol2')
        return

    try:
        molecule = read_molecule(input_path)
        molecule = Chem.RemoveAllHs(molecule)
        name = '.'.join(input_path.split('/')[-1].split('.')[:-1])
    except Exception as e:
        return f'Could not read the molecule: {e}'

    positions, one_hot, charges = parse_molecule(molecule, is_geom=ddpm.is_geom)
    fragment_mask = np.ones_like(charges)
    linker_mask = np.zeros_like(charges)
    anchor_flags = np.zeros_like(charges)
    if anchors is not None:
        for anchor in anchors.split(','):
            anchor_flags[int(anchor) - 1] = 1

    property_context = np.ones_like(charges) * property_val
    dataset = [{
        'uuid': '0',
        'name': '0',
        'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device),
        'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device),
        'charges': torch.tensor(charges, dtype=const.TORCH_FLOAT, device=device),
        'anchors': torch.tensor(anchor_flags, dtype=const.TORCH_FLOAT, device=device),
        'fragment_mask': torch.tensor(fragment_mask, dtype=const.TORCH_FLOAT, device=device),
        'linker_mask': torch.tensor(linker_mask, dtype=const.TORCH_FLOAT, device=device),
        'num_atoms': len(positions),
        'property_context': torch.tensor(property_context, dtype=const.TORCH_FLOAT, device=device)
    }] * n_samples
    
    batch_size = min(n_samples, 64) #TRY to make sure n_samples < 64
    dataloader = get_dataloader(dataset, batch_size=batch_size, collate_fn=collate_with_fragment_edges)

    # Sampling
    for batch_i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        print('Sampling only the FINAL output...') 
        chain, node_mask = ddpm.sample_chain(data, sample_fn=sample_fn, keep_frames=ddpm.FRAMES) #FBLD
        
        x = chain[0][:, :, :ddpm.n_dims]
        h = chain[0][:, :, ddpm.n_dims:]
        hsize = h.size()

        # Put the molecule back to the initial orientation
        com_mask = data['fragment_mask'] if ddpm.center_of_mass == 'fragments' else data['anchors']
        pos_masked = data['positions'] * com_mask
        N = com_mask.sum(1, keepdims=True)
        mean = torch.sum(pos_masked, dim=1, keepdim=True) / N
        x = x + mean * node_mask

        offset_idx = batch_i * batch_size
        names = [f'output_{offset_idx+i}_{name}' for i in range(batch_size)]
        save_xyz_file(output_dir, h, x, node_mask, names=names, is_geom=ddpm.is_geom, suffix='')

        for i in range(batch_size):
            out_xyz = f'{output_dir}/output_{offset_idx+i}_{name}_.xyz'
            out_sdf = f'{output_dir}/output_{offset_idx+i}_{name}_.sdf'
            subprocess.run(f'obabel {out_xyz} -O {out_sdf} 2> /dev/null', shell=True)
        print(f'Saved generated molecules in .xyz and .sdf format in directory {output_dir}')

        # chain_batch, node_mask = ddpm.sample_chain(data, sample_fn=sample_fn, keep_frames=ddpm.FRAMES) #ddpm.FRAMES = 100 default 
        chain_batch = chain
        #chain_batch : (F,B,L,D); node_mask : (B,D)

        print('Sampling ALL trajectories...')
        # Get final molecules from chains – for computing metrics
        #Below not necessary!?
        x, h = split_features(
            z=chain_batch[0], #0th frame --> (BLD)
            n_dims=ddpm.n_dims,
            num_classes=ddpm.num_classes,
            include_charges=ddpm.include_charges,
        )

        # # Save molecules without pockets
        # if '.' in ddpm.train_data_prefix:
        #     node_mask = node_mask - data['pocket_mask']

        #BELOW: Coordinate + BONDs + BOND ORDER!!!!!
        one_hot = h['categorical'] #from utils.split_features (BLC)
        # one_hot = h
        print(one_hot.size(), hsize)

        pred_molecules_batch, moleculse_stable = build_molecules(one_hot, x, node_mask, is_geom=ddpm.is_geom) #List[mol] (size Batch)
        out_sdf = f'{output_dir}/BO_kept_mols.sdf'
        with Chem.SDWriter(open(out_sdf, 'w')) as writer:
            for i, (mol, sta) in enumerate(zip(pred_molecules_batch, moleculse_stable)):
                if sta:
                    writer.write(mol) #Save to SDF with conformations and bond order! Size Batch
                else:
                    continue
            print(f"Out of {len(moleculse_stable)} molecules, {sum(moleculse_stable)} are stable per B.O. analysis!")

        #BELOW: ANIMATIONS!!!
        # Generate animation – will always do it for molecules with idx 0, 110 and 360
        if ddpm.samples_dir is not None:
            generate_animation(ddpm=ddpm, chain_batch=chain_batch, node_mask=node_mask)
        print(f'Saved generated molecules with B.O in .sdf format in directory {output_dir} and animation in {ddpm.samples_dir}')



if __name__ == '__main__':
    args = parser.parse_args()
    main(
        input_path=args.fragments,
        model=args.model,
        output_dir=args.output,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        linker_size=args.linker_size,
        anchors=args.anchors,
        samples_dir=args.samples_dir, #give it to DDPM
        # out_mol_path=args.out_mol_path #set it anew
        property_val=args.property_val
    )

    # python -W ignore difflinker_sample_and_analyze.py --fragments <YOUR_PATH> --model models/geom_difflinker.ckpt --linker_size models/geom_size_gnn.ckpt --output <YOUR_PATH> --samples_dir <YOUR_PATH>
