import re
import networkx as nx
import numpy as np
import logging
from itertools import chain
import warnings
from ase import neighborlist
from ase.geometry import get_distances
from pymatgen.io.cif import CifParser
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import bonds
from numpy import cross, eye
from numpy.linalg import norm, inv
from scipy.linalg import expm
from .atomic_data import metals
from .cif2system import PBC3DF_sym

logger = logging.getLogger(__name__)


def R(axis, theta):
    """
        returns a rotation matrix that rotates a vector around axis by angle theta
    """
    return expm(cross(eye(3), axis / norm(axis) * theta))


def M(vec1, vec2):
    """
        returns a rotation matrix that rotates vec1 onto vec2
    """
    ax = np.cross(vec1, vec2)

    if np.any(ax):  # need to check that the rotation axis has non-zero components

        ax_norm = ax / norm(np.cross(vec1, vec2))
        cos_ang = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
        ang = np.arccos(cos_ang)
        return R(ax_norm, ang)

    else:
        return eye(3)


PT = [
    'H',
    'He',
    'Li',
    'Be',
    'B',
    'C',
    'N',
    'O',
    'F',
    'Ne',
    'Na',
    'Mg',
    'Al',
    'Si',
    'P',
    'S',
    'Cl',
    'Ar',
    'K',
    'Ca',
    'Sc',
    'Ti',
    'V',
    'Cr',
    'Mn',
    'Fe',
    'Co',
    'Ni',
    'Cu',
    'Zn',
    'Ga',
    'Ge',
    'As',
    'Se',
    'Br',
    'Kr',
    'Rb',
    'Sr',
    'Y',
    'Zr',
    'Nb',
    'Mo',
    'Tc',
    'Ru',
    'Rh',
    'Pd',
    'Ag',
    'Cd',
    'In',
    'Sn',
    'Sb',
    'Te',
    'I',
    'Xe',
    'Cs',
    'Ba',
    'Hf',
    'Ta',
    'W',
    'Re',
    'Os',
    'Ir',
    'Pt',
    'Au',
    'Hg',
    'Tl',
    'Pb',
    'Bi',
    'Po',
    'At',
    'Rn',
    'Fr',
    'Ra',
    'La',
    'Ce',
    'Pr',
    'Nd',
    'Pm',
    'Sm',
    'Eu',
    'Gd',
    'Tb',
    'Dy',
    'Ho',
    'Er',
    'Tm',
    'Yb',
    'Lu',
    'Ac',
    'Th',
    'Pa',
    'U',
    'Np',
    'Pu',
    'Am',
    'Cm',
    'Bk',
    'Cf',
    'Es',
    'Fm',
    'Md',
    'No',
    'Lr',
    'FG',
    'X']


def nl(string):
    return re.sub('[^0-9]', '', string)


def flatten(L):
    return list(chain(*L))


def read_cif(self):
    """ reads cifs, returns both pymatgen structure and ase atoms object """

    cif = CifParser(self.filename)
    struct = cif.get_structures(primitive=False)[0]

    ase_atoms = AseAtomsAdaptor.get_atoms(struct)

    return struct, ase_atoms


def cif_read_pymatgen(filename, charges=False, coplanarity_tolerance=0.1):
    valencies = {'C': 4.0, 'Si': 4.0, 'Ge': 4.0, 'N': 3.0,
                 'P': 3.0, 'As': 3.0, 'Sb': 3.0, 'O': 2.0,
                 'S': 2.0, 'Se': 2.0, 'Te': 2.0, 'F': 1.0,
                 'Cl': 1.0, 'Br': 1.0, 'I': 1.0, 'H': 1.0,
                 'X': 1.0}

    bond_types = {0.5: 'S', 1.0: 'S', 1.5: 'A', 2.0: 'D', 3.0: 'T'}

    with open(filename, 'r') as f:
        f = f.read()
        f = filter(None, f.split('\n'))

    charge_list = []
    charge_switch = False

    for line in f:

        s = line.split()
        if '_atom_site_charge' in s:
            charge_switch = True
        if '_loop' in s:
            charge_switch = False

        if len(s) > 5:
            if charges:
                if charge_switch:
                    charge_list.append(float(s[-1]))

    cif = CifParser(filename)
    struct = cif.get_structures(primitive=False)[0]
    atoms = AseAtomsAdaptor.get_atoms(struct)
    unit_cell = atoms.get_cell()
    inv_uc = inv(unit_cell.T)
    elements = atoms.get_chemical_symbols()
    small_skin_metals = (
        'Ce',
        'Pr',
        'Nd',
        'Pm',
        'Sm',
        'Eu',
        'Gd',
        'Tb',
        'Dy',
        'Ho',
        'Er',
        'Tm',
        'Yb',
        'Lu',
        'Ba',
        'La')

    if any(i in elements for i in ('Zn')):
        skin = 0.30
    if any(i in elements for i in small_skin_metals):
        skin = 0.05
    else:
        skin = 0.20

    logger.debug('skin for bond calculation is', skin)

    if not charges:
        charge_list = [0.0 for a in atoms]

    cutoffs = neighborlist.natural_cutoffs(atoms)
    NL = neighborlist.NewPrimitiveNeighborList(
        cutoffs,
        use_scaled_positions=False,
        self_interaction=False,
        skin=skin)  # default atom cutoffs work well
    NL.build([True, True, True], unit_cell, atoms.get_positions())

    G = nx.Graph()
    for a in atoms:
        G.add_node(a.index, element_symbol=a.symbol, position=a.position)

    for i in atoms:

        nbors = NL.get_neighbors(i.index)[0]
        isym = i.symbol

        for j in nbors:

            jsym = atoms[j].symbol

            if (isym not in metals) and (jsym not in metals) and not any(
                    e == 'X' for e in [isym, jsym]):
                try:
                    bond = bonds.CovalentBond(struct[i.index], struct[j])
                    bond_order = bond.get_bond_order()
                except ValueError:
                    bond_order = 1.0
            elif (isym == 'X' or jsym == 'X') and (isym not in metals) and (jsym not in metals):
                bond_order = 1.0
            else:
                bond_order = 0.5

            bond_length = get_distances(
                i.position,
                p2=atoms[j].position,
                cell=unit_cell,
                pbc=[
                    True,
                    True,
                    True])
            bond_length = np.round(bond_length[1][0][0], 3)

            G.add_edge(
                i.index,
                j,
                bond_length=bond_length,
                bond_order=bond_order,
                bond_type='',
                pymatgen_bond_order=bond_order)

    NMG = G.copy()
    edge_list = list(NMG.edges())

    for e0, e1 in edge_list:

        sym0 = NMG.nodes[e0]['element_symbol']
        sym1 = NMG.nodes[e1]['element_symbol']

        if sym0 in metals or sym1 in metals:
            NMG.remove_edge(e0, e1)

    for i, data in G.nodes(data=True):

        isym = data['element_symbol']
        nbors = list(G.neighbors(i))
        nbor_symbols = [G.nodes[n]['element_symbol'] for n in nbors]
        nonmetal_nbor_symbols = [n for n in nbor_symbols if n not in metals]

        # remove C-M bonds if C is also bonded to carboxylate atoms, these are
        # almost always wrong
        for n, nsym in zip(nbors, nbor_symbols):
            if isym == 'C' and sorted(nonmetal_nbor_symbols) == ['C', 'O', 'O'] and nsym in metals:
                G.remove_edge(i, n)

    # intial bond typing, guessed from rounding pymatgen bond orders
    linkers = nx.connected_components(NMG)
    aromatic_atoms = []
    for linker in linkers:

        SG = NMG.subgraph(linker)

        for i, data in SG.nodes(data=True):

            isym = data['element_symbol']
            nbors = list(G.neighbors(i))
            nbor_symbols = [G.nodes[n]['element_symbol'] for n in nbors]
            nonmetal_nbor_symbols = [
                n for n in nbor_symbols if n not in metals]
            CB = nx.cycle_basis(SG)

            check_cycles = True
            if len(CB) < 3:
                check_cycles = False

            cyloc = None
            if check_cycles:
                for cy in range(len(CB)):
                    if i in CB[cy]:
                        cyloc = cy

            bond_orders = []
            for n, nsym in zip(nbors, nbor_symbols):

                edge_data = G[n][i]
                bond_order = edge_data['bond_order']

                if bond_order < 1.0 and bond_order != 0.5:
                    bond_order = 1.0
                # shortest observed single bond had order 1.321
                elif 1.00 <= bond_order < 1.33:
                    bond_order = 1.0
                elif 1.33 <= bond_order < 1.75:
                    bond_order = 1.5
                # bond orders tend to be on the high end for aromatic compounds
                elif 1.75 <= bond_order < 2.00:
                    bond_order = 1.5
                elif 2.00 <= bond_order < 3.00:
                    bond_order = round(bond_order)

                # bonds between two disparate cycles or cycles and non-cycles
                # should have order 1.0
                if check_cycles and cyloc is not None:
                    if n not in CB[cyloc]:
                        bond_order = 1.0

                if any(
                        i in metals for i in nbor_symbols) and isym == 'O' and nsym == 'C':
                    bond_order = 1.5

                if nsym in metals:
                    bond_order = 0.5

                if isym == 'C' and len(nbor_symbols) == 4:
                    bond_order = 1.0

                if isym == 'C' and sorted(nbor_symbols) == ['C', 'O', 'O']:
                    if nsym == 'C':
                        bond_order = 1.0
                    elif nsym == 'O':
                        bond_order = 1.5
                    else:
                        pass

                edge_data['bond_order'] = bond_order
                bond_orders.append(bond_order)
                edge_data['bond_type'] = bond_types[bond_order]

        all_cycles = nx.simple_cycles(nx.to_directed(SG))
        all_cycles = set([tuple(sorted(cy))
                          for cy in all_cycles if len(cy) > 4])

        # # assign aromatic bond orders as 1.5 (in most cases they will be already)
        for cycle in all_cycles:

            # rotate the ring normal vec onto the z-axis to determine
            # coplanarity
            coords = np.array([G.nodes[c]['position'] for c in cycle])
            fcoords = np.dot(inv_uc, coords.T).T
            anchor = fcoords[0]
            fcoords = np.array([vec - PBC3DF_sym(anchor, vec)[1]
                                for vec in fcoords])
            coords = np.dot(unit_cell.T, fcoords.T).T

            coords -= np.average(coords, axis=0)

            vec0 = coords[0]
            vec1 = coords[1]

            normal = np.cross(vec0, vec1)
            RZ = M(normal, np.array([0.0, 0.0, 1.0]))
            coords = np.dot(RZ, coords.T).T
            maxZ = max([abs(z) for z in coords[:, -1]])

            # if coplanar make all bond orders 1.5
            if maxZ < coplanarity_tolerance:

                aromatic_atoms.extend(list(cycle))
                cycle_subgraph = SG.subgraph(cycle)

                for e0, e1 in cycle_subgraph.edges():
                    G[e0][e1]['bond_order'] = 1.5

    for i, data in G.nodes(data=True):

        isym = data['element_symbol']
        bond_orders = [G[i][n]['bond_order'] for n in G.neighbors(i)]
        total_bond_order = np.sum(bond_orders)
        bond_orders = [str(o) for o in bond_orders]
        nbor_symbols = ' '.join([G.nodes[n]['element_symbol']
                                 for n in G.neighbors(i)])

        if isym not in metals and total_bond_order != valencies[isym]:
            message = ' '.join([str(isym),
                                'has total bond order',
                                str(total_bond_order),
                                'with neighbors',
                                nbor_symbols,
                                'and bond orders'] + bond_orders)
            warnings.warn(message)

    elems = atoms.get_chemical_symbols()
    names = [a.symbol + str(a.index) for a in atoms]
    cif_labels = [x.label for x in struct.sites]
    ccoords = atoms.get_positions()
    fcoords = atoms.get_scaled_positions()

    bond_list = []
    for e0, e1, data in G.edges(data=True):
        sym0 = G.nodes[e0]['element_symbol']
        sym1 = G.nodes[e1]['element_symbol']

        name0 = sym0 + str(e0)
        name1 = sym1 + str(e1)

        bond_list.append(
            [name0, name1, '.', data['bond_type'], data['bond_length']])

    A, B, C = unit_cell.lengths()
    alpha, beta, gamma = unit_cell.angles()

    return elems, names, cif_labels, ccoords, fcoords, charge_list, bond_list, (
        A, B, C, alpha, beta, gamma), np.asarray(unit_cell).T
