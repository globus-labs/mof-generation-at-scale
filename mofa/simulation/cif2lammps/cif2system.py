import re
import math
import numpy as np
import networkx as nx
import itertools
import datetime
import logging
from . import atomic_data
import functools
from random import choice
import warnings

from numpy.linalg import norm

logger = logging.getLogger(__name__)

metals = atomic_data.metals
mass_key = atomic_data.mass_key

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


def GCD(a, b):
    a = abs(a)
    b = abs(b)
    while a:
        a, b = b % a, a
    return b


def GCD_List(list):
    return functools.reduce(GCD, list)


def nn(string):
    return re.sub('[^a-zA-Z]', '', string)


def nl(string):
    return re.sub('[^0-9]', '', string)


def isfloat(value):
    """
        determines if a value is a float
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def iscoord(line):
    """
        identifies coordinates in CIFs
    """
    if nn(line[0]) in PT and line[1] in PT and False not in map(
            isfloat, line[2:5]):
        return True
    else:
        return False


def isbond(line):
    """
        identifies bonding in cifs
    """
    if nn(line[0]) in PT and nn(line[1]) in PT and isfloat(
            line[2]) and line[-1] in ('S', 'D', 'T', 'A'):
        return True
    else:
        return False


def PBC3DF_sym(vec1, vec2):
    """
        applies periodic boundary to distance between vec1 and vec2 (fractional coordinates)
    """
    dist = vec1 - vec2
    sym_dist = [
        (-1.0, dim - 1.0) if dim > 0.5 else (1.0, dim + 1.0)
        if dim < -0.5 else (0, dim) for dim in dist
    ]
    sym = np.array([s[0] for s in sym_dist])
    ndist = np.array([s[1] for s in sym_dist])

    return ndist, sym


def cif_read(filename, charges=False, add_Zr_bonds=False):
    with open(filename, 'r') as f:
        f = f.read()
        f = filter(None, f.split('\n'))

    names = []
    cif_labels = []
    elems = []
    fcoords = []
    charge_list = []
    bonds = []

    for line in f:
        s = line.split()
        if '_cell_length_a' in line:
            a = s[1]
        if '_cell_length_b' in line:
            b = s[1]
        if '_cell_length_c' in line:
            c = s[1]
        if '_cell_angle_alpha' in line:
            alpha = s[1]
        if '_cell_angle_beta' in line:
            beta = s[1]
        if '_cell_angle_gamma' in line:
            gamma = s[1]
        if iscoord(s):

            names.append(s[0])
            cif_labels.append(s[0])
            elems.append(s[1])

            fvec = np.array([np.round(float(v), 8) for v in s[2:5]])
            for dim in range(len(fvec)):
                if fvec[dim] < 0.0:
                    fvec[dim] += 1.0
                elif fvec[dim] > 1.0:
                    fvec[dim] -= 1.0

            fcoords.append(fvec)
            charge_list.append(float(s[-1]))

        if isbond(s):
            bonds.append((s[0], s[1], s[3], s[4], s[2]))

    pi = np.pi
    a, b, c, alpha, beta, gamma = map(float, (a, b, c, alpha, beta, gamma))
    ax = a
    ay = 0.0
    az = 0.0
    bx = b * np.cos(gamma * pi / 180.0)
    by = b * np.sin(gamma * pi / 180.0)
    bz = 0.0
    cx = c * np.cos(beta * pi / 180.0)
    cy = (c * b * np.cos(alpha * pi / 180.0) - bx * cx) / by
    cz = (c ** 2.0 - cx ** 2.0 - cy ** 2.0) ** 0.5
    unit_cell = np.asarray([[ax, ay, az], [bx, by, bz], [cx, cy, cz]]).T

    ccoords = []

    for LL in fcoords:
        vec = LL
        vec = np.dot(unit_cell, vec)
        ccoords.append(vec)

    fcoords = np.asarray(fcoords)
    ccoords = np.asarray(ccoords)
    charge_list = np.asarray(charge_list)
    net_charge = np.round(np.sum(charge_list), 3)

    if net_charge > 0.1 and charges:
        warnings.warn(
            'A potentially significant net charge of ' +
            str(net_charge) +
            ' is being removed')

    if len(charge_list) > 0:
        remove_net = choice(range(len(charge_list)))
        charge_list[remove_net] -= net_charge

    if add_Zr_bonds:
        count = 0
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):

                elemi = elems[i]
                elemj = elems[j]

                if elemi == 'Zr' and elemj == 'Zr':

                    ivec = fcoords[i]
                    jvec = fcoords[j]

                    dist = PBC3DF_sym(ivec, jvec)[0]
                    dist = norm(np.dot(unit_cell, dist))

                    if dist < 4.5:
                        count += 1
                        bonds.append(
                            [names[i], names[j], '.', 'S', np.round(dist, 3)])

        logger.debug(count, 'Zr-Zr bonds added...')

    return elems, names, cif_labels, ccoords, fcoords, charge_list, bonds, (
        a, b, c, alpha, beta, gamma), unit_cell


def initialize_system(
        filename,
        charges=False,
        small_molecule_cutoff=5,
        read_pymatgen=False):
    if not read_pymatgen:
        elems, names, cif_labels, ccoords, fcoords, charge_list, bonds, uc_params, unit_cell = cif_read(
            filename, charges=charges)
    else:
        from .pymatgen_cif2system import cif_read_pymatgen
        elems, names, cif_labels, ccoords, fcoords, charge_list, bonds, uc_params, unit_cell = cif_read_pymatgen(
            filename, charges=charges)

    A, B, C, alpha, beta, gamma = uc_params

    G = nx.Graph()
    index = 0
    index_key = {}

    for e, n, cif_label, cc, fc, charge in zip(
            elems, names, cif_labels, ccoords, fcoords, charge_list):
        index += 1

        G.add_node(index,
                   element_symbol=e,
                   mol_flag='1',
                   index=index,
                   force_field_type='',
                   cartesian_position=cc,
                   fractional_position=fc,
                   charge=charge,
                   replication=np.array([0.0,
                                         0.0,
                                         0.0]),
                   duplicated_version_of=None,
                   cif_label=cif_label)
        index_key[n] = index

    for b in bonds:

        dist, sym = PBC3DF_sym(G.nodes[index_key[b[0]]]['fractional_position'],
                               G.nodes[index_key[b[1]]]['fractional_position'])

        if np.any(sym):
            sym_code = '1_' + ''.join(map(str, map(int, sym + 5)))
        else:
            sym_code = '.'

        dist = np.linalg.norm(np.dot(unit_cell, dist))

        G.add_edge(index_key[b[0]],
                   index_key[b[1]],
                   sym_code=sym_code,
                   bond_type=b[3],
                   length=float(b[-1]))

    print_flag = False
    for e in G.edges(data=True):

        e0, e1, data = e
        nbors0 = list(G.neighbors(e0))
        nbors1 = list(G.neighbors(e1))
        nbors0_symbols = [G.nodes[nb]['element_symbol'] for nb in nbors0]
        nbors1_symbols = [G.nodes[nb]['element_symbol'] for nb in nbors1]
        es0 = G.nodes[e0]['element_symbol']
        es1 = G.nodes[e1]['element_symbol']
        bond_type = data['bond_type']

        # carboxylate oxygens should have aromatic bonds with C
        if len(nbors0_symbols) == 2 and es0 == 'O' and es1 == 'C' and any(
                i in metals for i in nbors0_symbols) and bond_type != 'A':
            print_flag = True
            data['bond_type'] = 'A'
        if len(nbors1_symbols) == 2 and es1 == 'O' and es0 == 'C' and any(
                i in metals for i in nbors1_symbols) and bond_type != 'A':
            print_flag = True
            data['bond_type'] = 'A'

        # nitro nitrogens should have aromatic bonds with O
        if es0 == 'N' and es1 == 'O' and sorted(
                nbors0_symbols) == ['C', 'O', 'O']:
            print_flag = True
            data['bond_type'] = 'A'
        if es1 == 'N' and es0 == 'O' and sorted(
                nbors1_symbols) == ['C', 'O', 'O']:
            print_flag = True
            data['bond_type'] = 'A'

    if print_flag:
        logger.debug('correcting bond type to aromatic for', filename)

    components = []
    SGS = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    for S in SGS:

        elems = [data['element_symbol'] for node, data in S.nodes(data=True)]
        comp_dict = dict((k, 0) for k in set(elems))
        for es in elems:
            comp_dict[es] += 1

        counts = GCD_List([comp_dict[e] for e in comp_dict])
        for es in comp_dict:
            comp_dict[es] = int(comp_dict[es] / float(counts))

        comp = tuple(
            sorted([(key, val) for key, val in comp_dict.items()], key=lambda x: x[0]))
        formula = ''.join([str(x) for es in comp for x in es])
        components.append((len(elems), formula, S))

    logger.debug(
        'there are',
        len(components),
        'components in the system with (  # atoms, formula unit):')
    SM = nx.Graph()
    framework = nx.Graph()

    for component in components:

        logger.debug('{:<6} {}'.format(component[0], component[1]))
        S = component[2]

        if len(S.nodes()) > small_molecule_cutoff:
            framework = nx.compose(framework, S)

        if len(S.nodes()) < small_molecule_cutoff:

            node_elems = [(n, data['element_symbol'])
                          for n, data in S.nodes(data=True)]
            sort_elems = sorted(node_elems, key=lambda x: x[1], reverse=True)
            sort_index = sorted(node_elems, key=lambda x: x[0], reverse=False)
            sort_key = dict((i[0], j[0])
                            for i, j in zip(sort_index, sort_elems))
            add_graph = nx.Graph()

            for node, elem in sort_elems:
                data = G.nodes[node]
                add_graph.add_node(sort_key[node], **data)

            for e0, e1, data in S.edges(data=True):
                add_graph.add_edge(sort_key[e0], sort_key[e1], **data)

            SM = nx.compose(SM, add_graph)

    index = 0
    frame_remap = {}
    for name, data in framework.nodes(data=True):
        index += 1
        frame_remap[name] = index
        data['index'] = index
    framework = nx.relabel_nodes(framework, frame_remap)

    sm_remap = {}
    for name, data in SM.nodes(data=True):
        index += 1
        sm_remap[name] = index
        data['index'] = index
    SM = nx.relabel_nodes(SM, sm_remap)

    return {
        'box': (
            A,
            B,
            C,
            alpha,
            beta,
            gamma),
        'graph': framework,
        'SM_graph': SM,
        'max_ind': index}


def duplicate_system(system, replications, small_molecule_cutoff=10):
    if replications == '1x1x1':
        return system

    G = system['graph']
    SMG = system['SM_graph']
    G = nx.compose(G, SMG)
    box = system['box']

    replications = list(map(int, replications.split('x')))
    replicated_box = (
        box[0] *
        replications[0],
        box[1] *
        replications[1],
        box[2] *
        replications[2],
        box[3],
        box[4],
        box[5])

    pi = np.pi
    a, b, c, alpha, beta, gamma = replicated_box
    ax = a
    ay = 0.0
    az = 0.0
    bx = b * np.cos(gamma * pi / 180.0)
    by = b * np.sin(gamma * pi / 180.0)
    bz = 0.0
    cx = c * np.cos(beta * pi / 180.0)
    cy = (c * b * np.cos(alpha * pi / 180.0) - bx * cx) / by
    cz = (c ** 2.0 - cx ** 2.0 - cy ** 2.0) ** 0.5
    unit_cell = np.asarray([[ax, ay, az], [bx, by, bz], [cx, cy, cz]]).T

    basis_vecs = [np.array([1, 0, 0]), np.array(
        [0, 1, 0]), np.array([0, 0, 1])]
    dim0 = [[np.array([0, 0, 0])]] + [[np.array([0, 0, 0])] + [basis_vecs[0]
                                                               for i in range(r + 1)] for r in range(replications[0] - 1)]
    dim1 = [[np.array([0, 0, 0])]] + [[np.array([0, 0, 0])] + [basis_vecs[1]
                                                               for i in range(r + 1)] for r in range(replications[1] - 1)]
    dim2 = [[np.array([0, 0, 0])]] + [[np.array([0, 0, 0])] + [basis_vecs[2]
                                                               for i in range(r + 1)] for r in range(replications[2] - 1)]

    dim0 = [np.sum([v for v in comb], axis=0) for comb in dim0]
    dim1 = [np.sum([v for v in comb], axis=0) for comb in dim1]
    dim2 = [np.sum([v for v in comb], axis=0) for comb in dim2]

    trans_vecs = [np.sum(comb, axis=0)
                  for comb in itertools.product(dim0, dim1, dim2)]
    trans_vecs = [v for v in trans_vecs if np.any(v)]

    logger.debug('The transformation vectors for the replication are:')
    for vec in trans_vecs:
        logger.debug(vec)
    logger.debug('...')

    if len(trans_vecs) != replications[0] * \
            replications[1] * replications[2] - 1:
        raise ValueError(
            'The number of transformation vectors in the replication is wrong somehow')

    NG = G.copy()
    edge_remove_list = []
    max_ind = max([d['index'] for n, d in G.nodes(data=True)])
    count = max_ind
    equivalency = dict((n, []) for n in G.nodes())

    for trans_vec in trans_vecs:

        for node, node_data in G.nodes(data=True):
            count += 1

            # this data stays the same
            element_symbol = node_data['element_symbol']
            charge = node_data['charge']
            cif_label = node_data['cif_label']

            # update index
            original_atom = node_data['index']
            new_index = count

            # update coordinates
            fvec = node_data['fractional_position']
            translated_fvec = fvec + trans_vec
            fvec = np.array([c / d for c, d in zip(fvec, replications)])
            translated_fvec = np.array(
                [c / d for c, d in zip(translated_fvec, replications)])
            NG.nodes[node]['fractional_position'] = fvec

            equivalency[original_atom].append(new_index)
            NG.add_node(new_index,
                        element_symbol=element_symbol,
                        mol_flag=1,
                        index=new_index,
                        force_field_type='',
                        cartesian_position=np.array([0.0,
                                                     0.0,
                                                     0.0]),
                        fractional_position=translated_fvec,
                        charge=charge,
                        duplicated_version_of=original_atom,
                        cif_label=cif_label)

    for node, data in NG.nodes(data=True):
        data['cartesian_position'] = np.dot(
            unit_cell, data['fractional_position'])

    for n0, n1, edge_data in G.edges(data=True):

        sym_code = edge_data['sym_code']
        bond_type = edge_data['bond_type']
        length = edge_data['length']

        fvec_n0 = NG.nodes[n0]['fractional_position']
        fvec_n1 = NG.nodes[n1]['fractional_position']

        for eq0 in equivalency[n0]:
            for eq1 in equivalency[n1]:

                fvec_eq0 = NG.nodes[eq0]['fractional_position']
                fvec_eq1 = NG.nodes[eq1]['fractional_position']

                dist_e0e1, sym_e0e1 = PBC3DF_sym(fvec_eq0, fvec_eq1)
                dist_e0e1 = np.linalg.norm(np.dot(unit_cell, dist_e0e1))

                dist_n0e1, sym_n0e1 = PBC3DF_sym(fvec_n0, fvec_eq1)
                dist_n0e1 = np.linalg.norm(np.dot(unit_cell, dist_n0e1))

                dist_e0n1, sym_e0n1 = PBC3DF_sym(fvec_eq0, fvec_n1)
                dist_e0n1 = np.linalg.norm(np.dot(unit_cell, dist_e0n1))

                dist_n0n1, sym_n0n1 = PBC3DF_sym(fvec_n0, fvec_n1)
                dist_n0n1 = np.linalg.norm(np.dot(unit_cell, dist_n0n1))

                if abs(dist_e0e1 - length) < 0.075:

                    if np.any(sym_e0e1):
                        sym_code = '1_' + \
                                   ''.join(map(str, map(int, sym_e0e1 + 5)))
                    else:
                        sym_code = '.'

                    NG.add_edge(
                        eq0,
                        eq1,
                        sym_code=sym_code,
                        bond_type=bond_type,
                        length=dist_e0e1)

                if abs(dist_n0e1 - length) < 0.075:

                    if np.any(sym_n0e1):
                        sym_code = '1_' + \
                                   ''.join(map(str, map(int, sym_n0e1 + 5)))
                    else:
                        sym_code = '.'

                    NG.add_edge(
                        n0,
                        eq1,
                        sym_code=sym_code,
                        bond_type=bond_type,
                        length=dist_n0e1)

                if abs(dist_e0n1 - length) < 0.075:

                    if np.any(sym_e0n1):
                        sym_code = '1_' + \
                                   ''.join(map(str, map(int, sym_e0n1 + 5)))
                    else:
                        sym_code = '.'

                    NG.add_edge(
                        eq0,
                        n1,
                        sym_code=sym_code,
                        bond_type=bond_type,
                        length=dist_n0e1)

                if abs(dist_n0n1 - length) > 0.075:
                    if (n0, n1) not in edge_remove_list:
                        edge_remove_list.append((n0, n1))

    for e in edge_remove_list:
        NG.remove_edge(e[0], e[1])

    components = []
    SGS = [NG.subgraph(c).copy() for c in nx.connected_components(NG)]
    for S in SGS:

        elems = [data['element_symbol'] for node, data in S.nodes(data=True)]
        comp_dict = dict((k, 0) for k in set(elems))
        for es in elems:
            comp_dict[es] += 1

        counts = GCD_List([comp_dict[e] for e in comp_dict])
        for es in comp_dict:
            comp_dict[es] = int(comp_dict[es] / float(counts))

        comp = tuple(
            sorted([(key, val) for key, val in comp_dict.items()], key=lambda x: x[0]))
        formula = ''.join([str(x) for es in comp for x in es])
        components.append((len(elems), formula, S))

    logger.debug(
        'there are',
        len(components),
        'components in the system with (  # atoms, formula unit):')
    SM = nx.Graph()
    framework = nx.Graph()
    for component in components:
        logger.debug('{:<6} {}'.format(component[0], component[1]))
        S = component[2]
        if len(S.nodes()) > small_molecule_cutoff:
            framework = nx.compose(framework, S)
        if len(S.nodes()) < small_molecule_cutoff:
            SM = nx.compose(SM, S)

    index = 0
    frame_remap = {}
    for name, data in framework.nodes(data=True):
        index += 1
        frame_remap[name] = index
        data['index'] = index
    framework = nx.relabel_nodes(framework, frame_remap)

    sm_remap = {}
    for name, data in SM.nodes(data=True):
        index += 1
        sm_remap[name] = index
        data['index'] = index
    SM = nx.relabel_nodes(SM, sm_remap)

    MI = max([data['index'] for n, data in NG.nodes(data=True)])

    return {
        'box': replicated_box,
        'graph': framework,
        'SM_graph': SM,
        'max_ind': MI}


def replication_determination(system, replication, cutoff):
    box = system['box']

    pi = np.pi
    a, b, c, alpha, beta, gamma = box
    ax = a
    ay = 0.0
    az = 0.0
    bx = b * np.cos(gamma * pi / 180.0)
    by = b * np.sin(gamma * pi / 180.0)
    bz = 0.0
    cx = c * np.cos(beta * pi / 180.0)
    cy = (c * b * np.cos(alpha * pi / 180.0) - bx * cx) / by
    cz = (c ** 2.0 - cx ** 2.0 - cy ** 2.0) ** 0.5

    avec = np.array([ax, ay, az])
    bvec = np.array([bx, by, bz])
    cvec = np.array([cx, cy, cz])

    thetac = np.arccos(np.dot(np.cross(avec, bvec), cvec) /
                       (np.linalg.norm(np.cross(avec, bvec)) *
                        np.linalg.norm(cvec)))
    dist2 = np.absolute(np.linalg.norm(cvec) * np.cos(thetac))

    thetab = np.arccos(np.dot(np.cross(avec, cvec), bvec) /
                       (np.linalg.norm(np.cross(avec, cvec)) *
                        np.linalg.norm(bvec)))
    dist1 = np.absolute(np.linalg.norm(bvec) * np.cos(thetab))

    thetaa = np.arccos(np.dot(np.cross(cvec, bvec), avec) /
                       (np.linalg.norm(np.cross(cvec, bvec)) *
                        np.linalg.norm(avec)))
    dist0 = np.absolute(np.linalg.norm(avec) * np.cos(thetaa))

    if 'min_atoms' in replication:

        min_atoms = int(replication.split(':')[-1])

        G = system['graph']
        Natoms = float(len(G.nodes()))
        dmin = int(math.ceil(min_atoms / Natoms))

        dsep0 = int(math.ceil((2 * cutoff) / dist0))
        dsep1 = int(math.ceil((2 * cutoff) / dist1))
        dsep2 = int(math.ceil((2 * cutoff) / dist2))
        dsep = dsep0 * dsep1 * dsep2

        duplications = max(dsep, dmin)
        useable_shapes = []

        logger.debug('minimum duplications allowed:', duplications)

        while len(useable_shapes) < 1:
            rvals = range(duplications + 1)[1:]
            shapes = itertools.product(rvals, rvals, rvals)
            shapes = [s for s in shapes if functools.reduce(
                (lambda x, y: x * y), s) == duplications]
            useable_shapes = [
                s for s in shapes if min(
                    dist0 * s[0],
                    dist1 * s[1],
                    dist2 * s[2]) >= 2 * cutoff]
            useable_shapes = [s for s in useable_shapes if max([((a * s[0]) / (b * s[1])),
                                                                ((a * s[0]) / (c * s[2])),
                                                                ((b * s[1]) / (c * s[2])),
                                                                ((b * s[1]) / (a * s[0])),
                                                                ((c * s[2]) / (a * s[0])),
                                                                ((c * s[2]) / (b * s[1]))]) <= 2.0]
            duplications += 1

        duplications -= 1
        logger.debug('final duplications:', duplications)
        logger.debug('final number of atoms:', int(duplications * Natoms))

        shape_deviations = [(i, np.std([useable_shapes[i][0] *
                                        a, useable_shapes[i][1] *
                                        b, useable_shapes[i][2] *
                                        c])) for i in range(len(useable_shapes))]
        shape_deviations.sort(key=lambda x: x[1])
        selected_shape = useable_shapes[shape_deviations[0][0]]

        replication = 'x'.join(map(str, selected_shape))
        logger.debug(
            'replicating to a',
            replication,
            'cell (' +
            str(duplications) +
            ' duplications)...')
        system = duplicate_system(system, replication)
        logger.debug('the minimum boundary-boundary distance is',
                     min([d * s for d, s in zip(selected_shape, (dist0, dist1, dist2))]))
        replication = 'ma' + str(min_atoms)

        a, b, c, alpha, beta, gamma = system['box']
        lx = np.round(a, 8)
        xy = np.round(b * np.cos(math.radians(gamma)), 8)
        xz = np.round(c * np.cos(math.radians(beta)), 8)
        ly = np.round(np.sqrt(b ** 2 - xy ** 2), 8)
        yz = np.round((b * c * np.cos(math.radians(alpha)) - xy * xz) / ly, 8)
        lz = np.round(np.sqrt(c ** 2 - xz ** 2 - yz ** 2), 8)

        logger.debug('lx =', np.round(lx, 3), '(dim 0 separation = ' +
                     str(np.round(selected_shape[0] * dist0, 3)) + ')')
        logger.debug('ly =', np.round(ly, 3), '(dim 1 separation = ' +
                     str(np.round(selected_shape[1] * dist1, 3)) + ')')
        logger.debug('lz =', np.round(lz, 3), '(dim 2 separation = ' +
                     str(np.round(selected_shape[2] * dist2, 3)) + ')')
        logger.debug('alpha =', np.round(alpha, 3))
        logger.debug('beta  =', np.round(beta, 3))
        logger.debug('gamma =', np.round(gamma, 3))

    elif 'cutoff' in replication:

        dsep0 = int(math.ceil((2 * cutoff) / dist0))
        dsep1 = int(math.ceil((2 * cutoff) / dist1))
        dsep2 = int(math.ceil((2 * cutoff) / dist2))
        dsep = dsep0 * dsep1 * dsep2

        duplications = dsep
        useable_shapes = []

        logger.debug('minimum duplications allowed:', duplications)

        while len(useable_shapes) < 1:
            rvals = range(duplications + 1)[1:]
            shapes = itertools.product(rvals, rvals, rvals)
            shapes = [s for s in shapes if functools.reduce(
                (lambda x, y: x * y), s) == duplications]
            useable_shapes = [
                s for s in shapes if min(
                    dist0 * s[0],
                    dist1 * s[1],
                    dist2 * s[2]) >= 2 * cutoff]
            duplications += 1

        duplications -= 1
        logger.debug('final duplications:', duplications)

        shape_deviations = [(i, np.std([useable_shapes[i][0] *
                                        a, useable_shapes[i][1] *
                                        b, useable_shapes[i][2] *
                                        c])) for i in range(len(useable_shapes))]
        shape_deviations.sort(key=lambda x: x[1])
        selected_shape = useable_shapes[shape_deviations[0][0]]

        replication = 'x'.join(map(str, selected_shape))
        logger.debug(
            'replicating to a',
            replication,
            'cell (' +
            str(duplications) +
            ' duplications)...')
        system = duplicate_system(system, replication)
        logger.debug('the minimum boundary-boundary distance is',
                     min([d * s for d, s in zip(selected_shape, (dist0, dist1, dist2))]))

        a, b, c, alpha, beta, gamma = system['box']
        lx = np.round(a, 8)
        xy = np.round(b * np.cos(math.radians(gamma)), 8)
        xz = np.round(c * np.cos(math.radians(beta)), 8)
        ly = np.round(np.sqrt(b ** 2 - xy ** 2), 8)
        yz = np.round((b * c * np.cos(math.radians(alpha)) - xy * xz) / ly, 8)
        lz = np.round(np.sqrt(c ** 2 - xz ** 2 - yz ** 2), 8)

        logger.debug('lx =', np.round(lx, 3), '(dim 0 separation = ' +
                     str(np.round(selected_shape[0] * dist0, 3)) + ')')
        logger.debug('ly =', np.round(ly, 3), '(dim 1 separation = ' +
                     str(np.round(selected_shape[1] * dist1, 3)) + ')')
        logger.debug('lz =', np.round(lz, 3), '(dim 2 separation = ' +
                     str(np.round(selected_shape[2] * dist2, 3)) + ')')
        logger.debug('alpha =', np.round(alpha, 3))
        logger.debug('beta  =', np.round(beta, 3))
        logger.debug('gamma =', np.round(gamma, 3))

    elif 'x' in replication and replication != '1x1x1':

        system = duplicate_system(system, replication)

    elif replication == '1x1x1':

        pass

    elif replication == '':

        pass

    else:

        raise ValueError('The replication command is not recognized')

    return system, replication


def write_cif_from_system(system, filename):
    box = system['box']
    G = system['graph']
    a, b, c, alpha, beta, gamma = box

    with open(filename, 'w') as out:
        out.write('data_' + filename[0:-4] + '\n')
        out.write('_audit_creation_date              ' +
                  datetime.datetime.today().strftime('%Y-%m-%d') + '\n')
        out.write("_audit_creation_method            'cif2lammps'" + '\n')
        out.write("_symmetry_space_group_name_H-M    'P1'" + '\n')
        out.write('_symmetry_Int_Tables_number       1' + '\n')
        out.write('_symmetry_cell_setting            triclinic' + '\n')
        out.write('loop_' + '\n')
        out.write('_symmetry_equiv_pos_as_xyz' + '\n')
        out.write('  x,y,z' + '\n')
        out.write('_cell_length_a                    ' + str(a) + '\n')
        out.write('_cell_length_b                    ' + str(b) + '\n')
        out.write('_cell_length_c                    ' + str(c) + '\n')
        out.write('_cell_angle_alpha                 ' + str(alpha) + '\n')
        out.write('_cell_angle_beta                  ' + str(beta) + '\n')
        out.write('_cell_angle_gamma                 ' + str(gamma) + '\n')
        out.write('loop_' + '\n')
        out.write('_atom_site_label' + '\n')
        out.write('_atom_site_type_symbol' + '\n')
        out.write('_atom_site_fract_x' + '\n')
        out.write('_atom_site_fract_y' + '\n')
        out.write('_atom_site_fract_z' + '\n')
        out.write('_atom_site_charge' + '\n')

        index_dict = {}
        for n, data in G.nodes(data=True):
            vec = data['fractional_position']
            es = data['element_symbol']
            ind = es + str(data['index'])
            index_dict[n] = ind
            chg = data['charge']
            out.write(
                '{:7} {:>4} {:>15.6f} {:>15.6f} {:>15.6f} {:>15.6f}'.format(
                    ind, es, vec[0], vec[1], vec[2], chg))
            out.write('\n')

        out.write('loop_' + '\n')
        out.write('_geom_bond_atom_site_label_1' + '\n')
        out.write('_geom_bond_atom_site_label_2' + '\n')
        out.write('_geom_bond_distance' + '\n')
        # out.write('_geom_bond_site_symmetry_1' + '\n')
        out.write('_ccdc_geom_bond_type' + '\n')

        for n0, n1, data in G.edges(data=True):
            ind0 = index_dict[n0]
            ind1 = index_dict[n1]
            dist = np.round(data['length'], 3)
            bond_type = data['bond_type']

            out.write(
                '{:7} {:>7} {:>7} {:>3}'.format(
                    ind0, ind1, dist, bond_type))
            out.write('\n')
