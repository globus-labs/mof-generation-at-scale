import gzip
import json
import logging
import os
import pickle
from typing import Union
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import const


def read_sdf(sdf_path):
    with Chem.SDMolSupplier(sdf_path, sanitize=False) as supplier:
        for molecule in supplier:
            yield molecule


def get_one_hot(atom, atoms_dict):
    one_hot = np.zeros(len(atoms_dict))
    one_hot[atoms_dict[atom]] = 1
    return one_hot


def parse_molecule(mol, is_geom):
    """Produce a 3D representation of a molecule given an RDKit molecule"""
    one_hot = []
    charges = []
    atom2idx = const.GEOM_ATOM2IDX if is_geom else const.ATOM2IDX
    charges_dict = const.GEOM_CHARGES if is_geom else const.CHARGES
    for atom in mol.GetAtoms():
        one_hot.append(get_one_hot(atom.GetSymbol(), atom2idx))
        charges.append(charges_dict[atom.GetSymbol()])
    positions = mol.GetConformer().GetPositions()
    return positions, np.array(one_hot), np.array(charges)


class MOFA_Dataset(Dataset):

    """Pytorch Dataset for MOFA generated data

    Example Usage: ds = MOFA_Dataset("datasets", "mof.json.gz", "cpu")
    """
    # Create a custom logger

    def __init__(self, data_path, prefix, device):
        self.log = logging.getLogger(__name__)
        if len(prefix.split(".")) != 1:
            self.log(
                f"WARNING the prefix you provided contains a period or an extension"
            )
            prefix = prefix.split(".")[0]

        dataset_path = os.path.join(data_path, f"{prefix}.pt")

        # if the data is already in pt format read it
        if os.path.exists(dataset_path):
            self.data = torch.load(dataset_path, map_location=device)
        # otherwise preprocess and save it in pt format
        else:
            print(f"Preprocessing dataset with prefix {prefix}")
            self.data = self.preprocess(data_path, prefix, device)
            torch.save(self.data, dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def preprocess(self, data_path, prefix, device):
        training_set = []
        # assumes since the data was not saved as pytorch its saved as json.gz
        dataset_path = os.path.join(data_path, f"{prefix}.json.gz")
        assert os.path.isfile(dataset_path)
        mofa_data = self._read_mofa_from_jsongz(dataset_path)
        for mofa_mof in mofa_data:
            training_set.extend(self._process_one_mof_result(mofa_mof, device))
        return training_set

    def _read_mofa_from_jsongz(self, filepath: Union[str, os.PathLike]):
        json_objects = []
        with gzip.open(filepath, "rt", encoding="utf-8") as file:
            for line in file:
                try:
                    json_object = json.loads(line)
                    json_objects.append(json_object)
                except json.JSONDecodeError as e:
                    self.log(f"Error decoding JSON of MOF object: {line}")
                    self.log(e)
        return json_objects

    def _process_one_mof_result(self, mofa_mof, device):
        """Formats the ligands in a MOFA_MOF into a training object for the difflinker training script

        Parameters:
        - mofa_mof: a mof that was generated within the mofa workflow

        Returns:
        - difflinker_ligand: a ligand in the format difflinker expects for training
        """
        ligands = mofa_mof.get("ligands", None)
        assert ligands, "There are no ligands in this MOFA_MOF"
        generated_training_samples = []
        # remove duplicates
        seen = set()

        for ligand in mofa_mof["ligands"]:
            # TODO: this was done by smile string which is not a good unique identifier
            # Tried the string representation non-hashable list, and tried converted training data into tuples (only the first ligand of each mof was accepted)
            identifier = ligand["smiles"]
            if identifier in seen:
                # TODO: investigate why printing here doesn't work
                self.log(
                    "MOFA Dataset to Training Set Conversion: Skipping ligand that we've already seen"
                )
                self.log(identifier)
                continue
            else:
                seen.add(identifier)
                training_ligand = {}
                training_ligand["uuid"] = str(uuid4())
                training_ligand["name"] = ligand["smiles"]
                xyz_info = ligand["xyz"].split("\n\n")[1]

                # grab position information removing atom types and hydrogens
                positions = [
                    line.split()[1:]
                    for line in xyz_info.strip().split("\n")
                    if line.split()[0] != "H"
                ]
                positions = np.array(positions, dtype=float)
                positions = torch.tensor(positions, dtype=torch.float, device=device)
                training_ligand["positions"] = positions

                # get the number of atoms with hydrogens removed and create one-hot encoding of atom types
                atom_types = [
                    line.split()[0]
                    for line in xyz_info.strip().split("\n")
                    if line.split()[0] != "H"
                ]
                atom_numbers = [
                    const.GEOM_ATOM2IDX[atom] for atom in atom_types if atom != "H"
                ]
                training_ligand["num_atoms"] = len(atom_numbers)
                atom_numbers_tensor = torch.tensor(
                    atom_numbers, dtype=torch.long, device=device
                )
                one_hot_encoded = F.one_hot(
                    atom_numbers_tensor, num_classes=len(const.GEOM_ATOM2IDX.keys())
                )
                training_ligand["one_hot"] = one_hot_encoded.float()

                # convert atom types to charges
                charges = [
                    const.GEOM_CHARGES[atom] for atom in atom_types if atom != "H"
                ]
                training_ligand["charges"] = torch.tensor(
                    charges, dtype=torch.float, device=device
                )

                # create linker_mask and fragment_mask using the "anchors" in the mofa data
                flattened_indices = {
                    index for sublist in ligand["anchor_atoms"] for index in sublist
                }
                linker_mask = [
                    0 if i in flattened_indices else 1 for i in range(len(atom_numbers))
                ]
                training_ligand["linker_mask"] = torch.tensor(
                    linker_mask, dtype=torch.float
                )
                training_ligand["fragment_mask"] = torch.tensor(
                    [1 if x == 0 else 0 for x in linker_mask],
                    dtype=torch.float,
                    device=device,
                )

                # create anchors for difflinker
                # TODO: this is fragile (its just the carbons in the two types we used COO and CN)
                anchors = (
                    np.array(atom_types)[training_ligand["fragment_mask"] == 1] == "C"
                )
                new_anchor_array = np.full(
                    len(atom_types), fill_value=False, dtype=bool
                )
                new_anchor_array[: min(len(anchors), len(atom_types))] = anchors
                training_ligand["anchors"] = torch.tensor(
                    new_anchor_array, device=device
                ).float()

                generated_training_samples.append(training_ligand)
        return generated_training_samples


class ZincDataset(Dataset):
    def __init__(self, data_path, prefix, device):
        dataset_path = os.path.join(data_path, f"{prefix}.pt")
        if os.path.exists(dataset_path):
            self.data = torch.load(dataset_path, map_location=device)
        else:
            print(f"Preprocessing dataset with prefix {prefix}")
            self.data = ZincDataset.preprocess(data_path, prefix, device)
            torch.save(self.data, dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def preprocess(data_path, prefix, device):
        data = []
        table_path = os.path.join(data_path, f"{prefix}_table.csv")
        fragments_path = os.path.join(data_path, f"{prefix}_frag.sdf")
        linkers_path = os.path.join(data_path, f"{prefix}_link.sdf")

        is_geom = ("geom" in prefix) or ("MOAD" in prefix)
        is_multifrag = "multifrag" in prefix

        table = pd.read_csv(table_path)
        generator = tqdm(
            zip(table.iterrows(), read_sdf(fragments_path), read_sdf(linkers_path)),
            total=len(table),
        )
        for (_, row), fragments, linker in generator:
            uuid = row["uuid"]
            name = row["molecule"]
            frag_pos, frag_one_hot, frag_charges = parse_molecule(
                fragments, is_geom=is_geom
            )
            link_pos, link_one_hot, link_charges = parse_molecule(
                linker, is_geom=is_geom
            )

            positions = np.concatenate([frag_pos, link_pos], axis=0)
            one_hot = np.concatenate([frag_one_hot, link_one_hot], axis=0)
            charges = np.concatenate([frag_charges, link_charges], axis=0)
            anchors = np.zeros_like(charges)

            if is_multifrag:
                for anchor_idx in map(int, row["anchors"].split("-")):
                    anchors[anchor_idx] = 1
            else:
                anchors[row["anchor_1"]] = 1
                anchors[row["anchor_2"]] = 1
            fragment_mask = np.concatenate(
                [np.ones_like(frag_charges), np.zeros_like(link_charges)]
            )
            linker_mask = np.concatenate(
                [np.zeros_like(frag_charges), np.ones_like(link_charges)]
            )

            data.append(
                {
                    "uuid": uuid,
                    "name": name,
                    "positions": torch.tensor(
                        positions, dtype=const.TORCH_FLOAT, device=device
                    ),
                    "one_hot": torch.tensor(
                        one_hot, dtype=const.TORCH_FLOAT, device=device
                    ),
                    "charges": torch.tensor(
                        charges, dtype=const.TORCH_FLOAT, device=device
                    ),
                    "anchors": torch.tensor(
                        anchors, dtype=const.TORCH_FLOAT, device=device
                    ),
                    "fragment_mask": torch.tensor(
                        fragment_mask, dtype=const.TORCH_FLOAT, device=device
                    ),
                    "linker_mask": torch.tensor(
                        linker_mask, dtype=const.TORCH_FLOAT, device=device
                    ),
                    "num_atoms": len(positions),
                }
            )

        return data


class MOADDataset(Dataset):
    def __init__(self, data_path, prefix, device):
        if "." in prefix:
            prefix, pocket_mode = prefix.split(".")
        else:
            parts = prefix.split("_")
            prefix = "_".join(parts[:-1])
            pocket_mode = parts[-1]

        dataset_path = os.path.join(data_path, f"{prefix}_{pocket_mode}.pt")
        if os.path.exists(dataset_path):
            self.data = torch.load(dataset_path, map_location=device)
        else:
            print(f"Preprocessing dataset with prefix {prefix}")
            self.data = self.preprocess(data_path, prefix, pocket_mode, device)
            torch.save(self.data, dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def preprocess(data_path, prefix, pocket_mode, device):
        data = []
        table_path = os.path.join(data_path, f"{prefix}_table.csv")
        fragments_path = os.path.join(data_path, f"{prefix}_frag.sdf")
        linkers_path = os.path.join(data_path, f"{prefix}_link.sdf")
        pockets_path = os.path.join(data_path, f"{prefix}_pockets.pkl")

        is_geom = True
        is_multifrag = "multifrag" in prefix

        with open(pockets_path, "rb") as f:
            pockets = pickle.load(f)

        table = pd.read_csv(table_path)
        generator = tqdm(
            zip(
                table.iterrows(),
                read_sdf(fragments_path),
                read_sdf(linkers_path),
                pockets,
            ),
            total=len(table),
        )
        for (_, row), fragments, linker, pocket_data in generator:
            uuid = row["uuid"]
            name = row["molecule"]
            frag_pos, frag_one_hot, frag_charges = parse_molecule(
                fragments, is_geom=is_geom
            )
            link_pos, link_one_hot, link_charges = parse_molecule(
                linker, is_geom=is_geom
            )

            # Parsing pocket data
            pocket_pos = pocket_data[f"{pocket_mode}_coord"]
            pocket_one_hot = []
            pocket_charges = []
            for atom_type in pocket_data[f"{pocket_mode}_types"]:
                pocket_one_hot.append(get_one_hot(atom_type, const.GEOM_ATOM2IDX))
                pocket_charges.append(const.GEOM_CHARGES[atom_type])
            pocket_one_hot = np.array(pocket_one_hot)
            pocket_charges = np.array(pocket_charges)

            positions = np.concatenate([frag_pos, pocket_pos, link_pos], axis=0)
            one_hot = np.concatenate(
                [frag_one_hot, pocket_one_hot, link_one_hot], axis=0
            )
            charges = np.concatenate(
                [frag_charges, pocket_charges, link_charges], axis=0
            )
            anchors = np.zeros_like(charges)

            if is_multifrag:
                for anchor_idx in map(int, row["anchors"].split("-")):
                    anchors[anchor_idx] = 1
            else:
                anchors[row["anchor_1"]] = 1
                anchors[row["anchor_2"]] = 1

            fragment_only_mask = np.concatenate(
                [
                    np.ones_like(frag_charges),
                    np.zeros_like(pocket_charges),
                    np.zeros_like(link_charges),
                ]
            )
            pocket_mask = np.concatenate(
                [
                    np.zeros_like(frag_charges),
                    np.ones_like(pocket_charges),
                    np.zeros_like(link_charges),
                ]
            )
            linker_mask = np.concatenate(
                [
                    np.zeros_like(frag_charges),
                    np.zeros_like(pocket_charges),
                    np.ones_like(link_charges),
                ]
            )
            fragment_mask = np.concatenate(
                [
                    np.ones_like(frag_charges),
                    np.ones_like(pocket_charges),
                    np.zeros_like(link_charges),
                ]
            )

            data.append(
                {
                    "uuid": uuid,
                    "name": name,
                    "positions": torch.tensor(
                        positions, dtype=const.TORCH_FLOAT, device=device
                    ),
                    "one_hot": torch.tensor(
                        one_hot, dtype=const.TORCH_FLOAT, device=device
                    ),
                    "charges": torch.tensor(
                        charges, dtype=const.TORCH_FLOAT, device=device
                    ),
                    "anchors": torch.tensor(
                        anchors, dtype=const.TORCH_FLOAT, device=device
                    ),
                    "fragment_only_mask": torch.tensor(
                        fragment_only_mask, dtype=const.TORCH_FLOAT, device=device
                    ),
                    "pocket_mask": torch.tensor(
                        pocket_mask, dtype=const.TORCH_FLOAT, device=device
                    ),
                    "fragment_mask": torch.tensor(
                        fragment_mask, dtype=const.TORCH_FLOAT, device=device
                    ),
                    "linker_mask": torch.tensor(
                        linker_mask, dtype=const.TORCH_FLOAT, device=device
                    ),
                    "num_atoms": len(positions),
                }
            )

        return data

    @staticmethod
    def create_edges(positions, fragment_mask_only, linker_mask_only):
        ligand_mask = fragment_mask_only.astype(bool) | linker_mask_only.astype(bool)
        ligand_adj = ligand_mask[:, None] & ligand_mask[None, :]
        proximity_adj = (
            np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1) <= 6
        )
        full_adj = ligand_adj | proximity_adj
        full_adj &= ~np.eye(len(positions)).astype(bool)

        curr_rows, curr_cols = np.where(full_adj)
        return [curr_rows, curr_cols]


class OptimisedMOADDataset(MOADDataset):
    # TODO: finish testing

    def __len__(self):
        return len(self.data["fragmentation_level_data"])

    def __getitem__(self, item):
        fragmentation_level_data = self.data["fragmentation_level_data"][item]
        protein_level_data = self.data["protein_level_data"][
            fragmentation_level_data["name"]
        ]
        return {
            **fragmentation_level_data,
            **protein_level_data,
        }

    @staticmethod
    def preprocess(data_path, prefix, pocket_mode, device):
        print("Preprocessing optimised version of the dataset")
        protein_level_data = {}
        fragmentation_level_data = []

        table_path = os.path.join(data_path, f"{prefix}_table.csv")
        fragments_path = os.path.join(data_path, f"{prefix}_frag.sdf")
        linkers_path = os.path.join(data_path, f"{prefix}_link.sdf")
        pockets_path = os.path.join(data_path, f"{prefix}_pockets.pkl")

        is_geom = True
        is_multifrag = "multifrag" in prefix

        with open(pockets_path, "rb") as f:
            pockets = pickle.load(f)

        table = pd.read_csv(table_path)
        generator = tqdm(
            zip(
                table.iterrows(),
                read_sdf(fragments_path),
                read_sdf(linkers_path),
                pockets,
            ),
            total=len(table),
        )
        for (_, row), fragments, linker, pocket_data in generator:
            uuid = row["uuid"]
            name = row["molecule"]
            frag_pos, frag_one_hot, frag_charges = parse_molecule(
                fragments, is_geom=is_geom
            )
            link_pos, link_one_hot, link_charges = parse_molecule(
                linker, is_geom=is_geom
            )

            # Parsing pocket data
            pocket_pos = pocket_data[f"{pocket_mode}_coord"]
            pocket_one_hot = []
            pocket_charges = []
            for atom_type in pocket_data[f"{pocket_mode}_types"]:
                pocket_one_hot.append(get_one_hot(atom_type, const.GEOM_ATOM2IDX))
                pocket_charges.append(const.GEOM_CHARGES[atom_type])
            pocket_one_hot = np.array(pocket_one_hot)
            pocket_charges = np.array(pocket_charges)

            positions = np.concatenate([frag_pos, pocket_pos, link_pos], axis=0)
            one_hot = np.concatenate(
                [frag_one_hot, pocket_one_hot, link_one_hot], axis=0
            )
            charges = np.concatenate(
                [frag_charges, pocket_charges, link_charges], axis=0
            )
            anchors = np.zeros_like(charges)

            if is_multifrag:
                for anchor_idx in map(int, row["anchors"].split("-")):
                    anchors[anchor_idx] = 1
            else:
                anchors[row["anchor_1"]] = 1
                anchors[row["anchor_2"]] = 1

            fragment_only_mask = np.concatenate(
                [
                    np.ones_like(frag_charges),
                    np.zeros_like(pocket_charges),
                    np.zeros_like(link_charges),
                ]
            )
            pocket_mask = np.concatenate(
                [
                    np.zeros_like(frag_charges),
                    np.ones_like(pocket_charges),
                    np.zeros_like(link_charges),
                ]
            )
            linker_mask = np.concatenate(
                [
                    np.zeros_like(frag_charges),
                    np.zeros_like(pocket_charges),
                    np.ones_like(link_charges),
                ]
            )
            fragment_mask = np.concatenate(
                [
                    np.ones_like(frag_charges),
                    np.ones_like(pocket_charges),
                    np.zeros_like(link_charges),
                ]
            )

            fragmentation_level_data.append(
                {
                    "uuid": uuid,
                    "name": name,
                    "anchors": torch.tensor(
                        anchors, dtype=const.TORCH_FLOAT, device=device
                    ),
                    "fragment_only_mask": torch.tensor(
                        fragment_only_mask, dtype=const.TORCH_FLOAT, device=device
                    ),
                    "pocket_mask": torch.tensor(
                        pocket_mask, dtype=const.TORCH_FLOAT, device=device
                    ),
                    "fragment_mask": torch.tensor(
                        fragment_mask, dtype=const.TORCH_FLOAT, device=device
                    ),
                    "linker_mask": torch.tensor(
                        linker_mask, dtype=const.TORCH_FLOAT, device=device
                    ),
                }
            )
            protein_level_data[name] = {
                "positions": torch.tensor(
                    positions, dtype=const.TORCH_FLOAT, device=device
                ),
                "one_hot": torch.tensor(
                    one_hot, dtype=const.TORCH_FLOAT, device=device
                ),
                "charges": torch.tensor(
                    charges, dtype=const.TORCH_FLOAT, device=device
                ),
                "num_atoms": len(positions),
            }

        return {
            "fragmentation_level_data": fragmentation_level_data,
            "protein_level_data": protein_level_data,
        }


def collate(batch):
    out = {}

    # Filter out big molecules
    # if 'pocket_mask' not in batch[0].keys():
    #    batch = [data for data in batch if data['num_atoms'] <= 50]
    # else:
    #    batch = [data for data in batch if data['num_atoms'] <= 1000]

    for i, data in enumerate(batch):
        for key, value in data.items():
            out.setdefault(key, []).append(value)

    for key, value in out.items():
        if key in const.DATA_LIST_ATTRS:
            continue
        if key in const.DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(
                value, batch_first=True, padding_value=0
            )
            continue
        # raise Exception(f'Unknown batch key: {key}')
        # print(f"Passing {key}")

    atom_mask = (out["fragment_mask"].bool() | out["linker_mask"].bool()).to(
        const.TORCH_INT
    )
    out["atom_mask"] = atom_mask[:, :, None]

    batch_size, n_nodes = atom_mask.size()

    # In case of MOAD edge_mask is batch_idx
    if "pocket_mask" in batch[0].keys():
        batch_mask = torch.cat(
            [torch.ones(n_nodes, dtype=const.TORCH_INT) * i for i in range(batch_size)]
        ).to(atom_mask.device)
        out["edge_mask"] = batch_mask
    else:
        edge_mask = atom_mask[:, None, :] * atom_mask[:, :, None]
        diag_mask = ~torch.eye(
            edge_mask.size(1), dtype=const.TORCH_INT, device=atom_mask.device
        ).unsqueeze(0)
        edge_mask *= diag_mask
        out["edge_mask"] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]

    return out


def collate_with_fragment_edges(batch):
    out = {}

    # Filter out big molecules
    # batch = [data for data in batch if data['num_atoms'] <= 50]

    for i, data in enumerate(batch):
        for key, value in data.items():
            out.setdefault(key, []).append(value)

    for key, value in out.items():
        if key in const.DATA_LIST_ATTRS:
            continue
        if key in const.DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(
                value, batch_first=True, padding_value=0
            )
            continue
        raise Exception(f"Unknown batch key: {key}")

    frag_mask = out["fragment_mask"]
    edge_mask = frag_mask[:, None, :] * frag_mask[:, :, None]
    diag_mask = ~torch.eye(
        edge_mask.size(1), dtype=const.TORCH_INT, device=frag_mask.device
    ).unsqueeze(0)
    edge_mask *= diag_mask

    batch_size, n_nodes = frag_mask.size()
    out["edge_mask"] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    # Building edges and covalent bond values
    rows, cols = [], []
    for batch_idx in range(batch_size):
        for i in range(n_nodes):
            for j in range(n_nodes):
                rows.append(i + batch_idx * n_nodes)
                cols.append(j + batch_idx * n_nodes)

    edges = [
        torch.LongTensor(rows).to(frag_mask.device),
        torch.LongTensor(cols).to(frag_mask.device),
    ]
    out["edges"] = edges

    atom_mask = (out["fragment_mask"].bool() | out["linker_mask"].bool()).to(
        const.TORCH_INT
    )
    out["atom_mask"] = atom_mask[:, :, None]

    for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]

    return out


def get_dataloader(dataset, batch_size, collate_fn=collate, shuffle=False):
    return DataLoader(dataset, batch_size, collate_fn=collate_fn, shuffle=shuffle)


def create_template(tensor, fragment_size, linker_size, fill=0):
    values_to_keep = tensor[:fragment_size]
    values_to_add = torch.ones(
        linker_size,
        tensor.shape[1],
        dtype=values_to_keep.dtype,
        device=values_to_keep.device,
    )
    values_to_add = values_to_add * fill
    return torch.cat([values_to_keep, values_to_add], dim=0)


def create_templates_for_linker_generation(data, linker_sizes):
    """
    Takes data batch and new linker size and returns data batch where fragment-related data is the same
    but linker-related data is replaced with zero templates with new linker sizes
    """
    decoupled_data = []
    for i, linker_size in enumerate(linker_sizes):
        data_dict = {}
        fragment_mask = data["fragment_mask"][i].squeeze()
        fragment_size = fragment_mask.sum().int()
        for k, v in data.items():
            if k == "num_atoms":
                # Computing new number of atoms (fragment_size + linker_size)
                data_dict[k] = fragment_size + linker_size
                continue
            if k in const.DATA_LIST_ATTRS:
                # These attributes are written without modification
                data_dict[k] = v[i]
                continue
            if k in const.DATA_ATTRS_TO_PAD:
                # Should write fragment-related data + (zeros x linker_size)
                fill_value = 1 if k == "linker_mask" else 0
                template = create_template(
                    v[i], fragment_size, linker_size, fill=fill_value
                )
                if k in const.DATA_ATTRS_TO_ADD_LAST_DIM:
                    template = template.squeeze(-1)
                data_dict[k] = template

        decoupled_data.append(data_dict)

    return collate(decoupled_data)
