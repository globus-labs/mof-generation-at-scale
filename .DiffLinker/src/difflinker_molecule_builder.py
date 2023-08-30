import torch
import numpy as np

from rdkit import Chem, Geometry

from src import const


def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer


def build_molecules(one_hot, x, node_mask, is_geom, margins=const.MARGINS_EDM):
    molecules = []
    stabilities = []
    for i in range(len(one_hot)):
        mask = node_mask[i].squeeze() == 1
        atom_types = one_hot[i][mask].argmax(dim=1).detach().cpu()
        positions = x[i][mask].detach().cpu()
        mol = build_molecule(positions, atom_types, is_geom, margins=margins)
        molecules.append(mol)
        molecule_stable, *_ = check_stability(positions, atom_types, is_geom, debug=False)
        stabilities.append(molecule_stable)
    return molecules, stabilities


def build_molecule(positions, atom_types, is_geom, margins=const.MARGINS_EDM):
    idx2atom = const.GEOM_IDX2ATOM if is_geom else const.IDX2ATOM
    X, A, E = build_xae_molecule(positions, atom_types, is_geom=is_geom, margins=margins)
    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(idx2atom[atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), const.BOND_DICT[E[bond[0], bond[1]].item()])

    mol.AddConformer(create_conformer(positions.detach().cpu().numpy().astype(np.float64)))
    return mol


def build_xae_molecule(positions, atom_types, is_geom, margins=const.MARGINS_EDM):
    """ Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3  (already masked to keep final number nodes)
        atom_types: N
        returns:
        X: N         (int)
        A: N x N     (bool) (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)

    idx2atom = const.GEOM_IDX2ATOM if is_geom else const.IDX2ATOM

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    for i in range(n):
        for j in range(i):

            pair = sorted([atom_types[i], atom_types[j]])
            order = get_bond_order(idx2atom[pair[0].item()], idx2atom[pair[1].item()], dists[i, j], margins=margins)

            # TODO: a batched version of get_bond_order to avoid the for loop
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order

    return X, A, E


def get_bond_order(atom1, atom2, distance, check_exists=True, margins=const.MARGINS_EDM):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in const.BONDS_1:
            return 0
        if atom2 not in const.BONDS_1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of the QM9 true samples
    if distance < const.BONDS_1[atom1][atom2] + margins[0]:

        # Check if atoms in bonds2 dictionary.
        if atom1 in const.BONDS_2 and atom2 in const.BONDS_2[atom1]:
            thr_bond2 = const.BONDS_2[atom1][atom2] + margins[1]
            if distance < thr_bond2:
                if atom1 in const.BONDS_3 and atom2 in const.BONDS_3[atom1]:
                    thr_bond3 = const.BONDS_3[atom1][atom2] + margins[2]
                    if distance < thr_bond3:
                        return 3  # Triple
                return 2  # Double
        return 1  # Single
    return 0  # No bond


############################
# Validity and bond analysis
def check_stability(positions, atom_type, is_geom, debug=False, margins=const.MARGINS_EDM):
    idx2atom = const.GEOM_IDX2ATOM if is_geom else const.IDX2ATOM

    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            pair = sorted([atom_type[i], atom_type[j]])

            order = get_bond_order(idx2atom[pair[0].item()], idx2atom[pair[1].item()], dist, margins=margins)

            nr_bonds[i] += order
            nr_bonds[j] += order
    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        # print(pair[0].item(), atom_type_i)

        possible_bonds = const.ALLOWED_BONDS[idx2atom[atom_type_i.item()]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (idx2atom[atom_type_i], nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return molecule_stable, nr_stable_bonds, len(x)

def analyze_stability_for_molecules(molecule_list, dataset_info):
    one_hot = molecule_list['one_hot']
    x = molecule_list['x']
    node_mask = molecule_list['node_mask']

    if isinstance(node_mask, torch.Tensor):
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [torch.sum(m) for m in node_mask]

    n_samples = len(x)

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0

    processed_list = []

    for i in range(n_samples):
        atom_type = one_hot[i].argmax(1).cpu().detach()
        pos = x[i].cpu().detach()

        atom_type = atom_type[0:int(atomsxmol[i])]
        pos = pos[0:int(atomsxmol[i])]
        processed_list.append((pos, atom_type))

    for mol in processed_list:
        pos, atom_type = mol
        validity_results = check_stability(pos, atom_type, dataset_info)

        molecule_stable += int(validity_results[0])
        nr_stable_bonds += int(validity_results[1])
        n_atoms += int(validity_results[2])

    # Validity
    fraction_mol_stable = molecule_stable / float(n_samples)
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
    }

    return validity_dict, None
