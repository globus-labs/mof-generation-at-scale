"""Functions which generate new ligands with DiffLinker"""
from functools import lru_cache
from typing import Iterator

import torch
import numpy as np
from rdkit import Chem

from mofa.model import LigandTemplate
from mofa.utils.src import const
from mofa.utils.src.datasets import collate_with_fragment_edges, get_dataloader, get_one_hot
from mofa.utils.src.lightning import DDPM
from mofa.utils.src.linker_size_lightning import SizeClassifier

DiffLinkerOutput = tuple[LigandTemplate, list[str], np.ndarray]
"""Output from a DiffLinker inference: the template used as a prompt, chemical symbols of selected types, coordinates of atoms"""


def read_molecules(path):
    if path.endswith('.pdb'):
        return Chem.MolFromPDBFile(path, sanitize=False, removeHs=True)
    elif path.endswith('.mol'):
        return Chem.MolFromMolFile(path, sanitize=False, removeHs=True)
    elif path.endswith('.mol2'):
        return Chem.MolFromMol2File(path, sanitize=False, removeHs=True)
    elif path.endswith('.sdf'):
        return Chem.SDMolSupplier(path, sanitize=False, removeHs=True)
    raise Exception('Unknown file extension')


@lru_cache(maxsize=1)  # Keep only one model in memory
def load_model(path, device) -> DDPM:
    """Load the DDPM model from disk"""
    return DDPM.load_from_checkpoint(path, map_location='cpu').eval().to(device)


def main_run(templates: list[LigandTemplate],
             model,
             output_dir,
             n_samples,
             n_steps,
             linker_size,
             device: str = 'cpu') -> Iterator[DiffLinkerOutput]:
    """Run the linker generation"""
    if linker_size.isdigit():
        linker_size = int(linker_size)

        def sample_fn(_data):
            return torch.ones(_data['positions'].shape[0], device=device, dtype=const.TORCH_INT) * linker_size
    else:
        size_nn = SizeClassifier.load_from_checkpoint(linker_size, map_location=device).eval().to(device)

        def sample_fn(_data):
            out, _ = size_nn.forward(_data, return_loss=False)
            probabilities = torch.softmax(out, dim=1)
            distribution = torch.distributions.Categorical(probs=probabilities)
            samples = distribution.sample()
            sizes = []
            for label in samples.detach().cpu().numpy():
                sizes.append(size_nn.linker_id2size[label])
            sizes = torch.tensor(sizes, device=samples.device, dtype=const.TORCH_INT)
            return sizes

    # Pull the model from disk, evicting the old one if needed
    ddpm = load_model(model, device)

    if n_steps is not None:
        ddpm.edm.T = n_steps  # otherwise, ddpm.edm.T = 1000 default

    # Get the lookup tables for atom types
    atom2idx = const.GEOM_ATOM2IDX if ddpm.is_geom else const.ATOM2IDX
    idx2atom = const.GEOM_IDX2ATOM if ddpm.is_geom else const.IDX2ATOM
    charges_dict = const.GEOM_CHARGES if ddpm.is_geom else const.CHARGES

    # Reading input fragments
    for n_mol, template in enumerate(templates):
        # Prepare the inputs for this structure
        symbols, positions, anchors = template.prepare_inputs()
        if ddpm.center_of_mass == 'anchors' and anchors is None:
            raise ValueError(
                'Please pass anchor atoms indices '
                'or use another DiffLinker model that does not require information about anchors'
            )

        one_hot = np.array([get_one_hot(s, atom2idx) for s in symbols])
        charges = np.array([charges_dict[s] for s in symbols])
        fragment_mask = np.ones_like(charges)
        linker_mask = np.zeros_like(charges)
        anchor_flags = np.zeros_like(charges)
        if anchors is not None:
            anchor_flags[anchors] = 1

        # Perform the sampling
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
        }] * n_samples
        batch_size = min(n_samples, 64)  # TRY to make sure n_samples < 64
        dataloader = get_dataloader(dataset, batch_size=batch_size, collate_fn=collate_with_fragment_edges)

        # Sampling
        for batch_i, data in enumerate(dataloader):
            chain, node_mask = ddpm.sample_chain(data, sample_fn=sample_fn, keep_frames=1)
            x = chain[0][:, :, :ddpm.n_dims]
            h = chain[0][:, :, ddpm.n_dims:]

            # Put the molecule back to the initial orientation
            com_mask = data['fragment_mask'] if ddpm.center_of_mass == 'fragments' else data['anchors']
            pos_masked = data['positions'] * com_mask
            N = com_mask.sum(1, keepdims=True)
            mean = torch.sum(pos_masked, dim=1, keepdim=True) / N
            x = x + mean * node_mask

            # Write out each generated structure
            batch_idx_selections = torch.argmax(h, dim=-1).detach().cpu().numpy()
            batch_coordinates = x.detach().cpu().numpy()
            for i in range(batch_size):
                # Convert the atom types to from one-hot to atomic numbers/symbols
                atom_types = [idx2atom[i] for i in batch_idx_selections[i, :]]

                # Make the output
                yield template, atom_types, batch_coordinates[i, :, :]
