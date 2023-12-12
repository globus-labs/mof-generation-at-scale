import networkx as nx
import numpy as np
import logging
from . import atomic_data
from itertools import groupby, combinations
from random import randint
from . import small_molecule_constants
from .cif2system import PBC3DF_sym
from . import write_molecule_files as WMF
from ase import Atom, Atoms
from ase import neighborlist
from ase.geometry import get_distances
from ase.io import read

mass_key = atomic_data.mass_key

logger = logging.getLogger(__name__)


def add_small_molecules(FF, ff_string):
    if ff_string == 'TraPPE':
        SM_constants = small_molecule_constants.TraPPE
    elif ff_string == 'TIP4P_2005_long':
        SM_constants = small_molecule_constants.TIP4P_2005_long
        FF.pair_data['special_bonds'] = 'lj 0.0 0.0 1.0 coul 0.0 0.0 0.0'
    elif ff_string == 'TIP4P_cutoff':
        SM_constants = small_molecule_constants.TIP4P_cutoff
        FF.pair_data['special_bonds'] = 'lj/coul 0.0 0.0 1.0'
    elif ff_string == 'TIP4P_2005_cutoff':
        SM_constants = small_molecule_constants.TIP4P_cutoff
        FF.pair_data['special_bonds'] = 'lj/coul 0.0 0.0 1.0'
    elif ff_string == 'Ions':
        SM_constants = small_molecule_constants.Ions
        FF.pair_data['special_bonds'] = 'lj/coul 0.0 0.0 1.0'
    # insert more force fields here if needed
    else:
        raise ValueError(
            'the small molecule force field',
            ff_string,
            'is not defined')

    SG = FF.system['graph']
    SMG = FF.system['SM_graph']

    if len(SMG.nodes()) > 0 and len(SMG.edges()) == 0:

        logger.debug('there are no small molecule bonds in the CIF, calculating based on covalent radii...')
        atoms = Atoms()

        offset = min(SMG.nodes())

        for node, data in SMG.nodes(data=True):
            # logger.debug(node, data)
            atoms.append(
                Atom(
                    data['element_symbol'],
                    data['cartesian_position']))

        atoms.set_cell(FF.system['box'])
        unit_cell = atoms.get_cell()
        cutoffs = neighborlist.natural_cutoffs(atoms)
        NL = neighborlist.NewPrimitiveNeighborList(
            cutoffs,
            use_scaled_positions=False,
            self_interaction=False,
            skin=0.10)  # shorten the cutoff a bit
        NL.build([True, True, True], unit_cell, atoms.get_positions())

        for i in atoms:

            nbors = NL.get_neighbors(i.index)[0]

            for j in nbors:
                bond_length = get_distances(
                    i.position,
                    p2=atoms[j].position,
                    cell=unit_cell,
                    pbc=[
                        True,
                        True,
                        True])
                bond_length = np.round(bond_length[1][0][0], 3)
                SMG.add_edge(
                    i.index + offset,
                    j + offset,
                    bond_length=bond_length,
                    bond_order='1.0',
                    bond_type='S')

        NMOL = len(list(nx.connected_components(SMG)))
        logger.debug(NMOL, 'small molecules were recovered after bond calculation')

    mol_flag = 1
    max_ind = FF.system['max_ind']
    index = max_ind

    box = FF.system['box']
    a, b, c, alpha, beta, gamma = box
    pi = np.pi
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
    inv_unit_cell = np.linalg.inv(unit_cell)

    add_nodes = []
    add_edges = []
    comps = []

    for comp in nx.connected_components(SMG):

        mol_flag += 1
        comp = sorted(list(comp))
        ID_string = sorted([SMG.nodes[n]['element_symbol'] for n in comp])
        ID_string = [(key, len(list(group)))
                     for key, group in groupby(ID_string)]
        ID_string = ''.join([str(e) for c in ID_string for e in c])
        comps.append(ID_string)

        for n in comp:

            data = SMG.nodes[n]

            SMG.nodes[n]['mol_flag'] = str(mol_flag)

            if ID_string == 'H2O1':
                SMG.nodes[n]['force_field_type'] = SMG.nodes[n]['element_symbol'] + '_w'
            else:
                SMG.nodes[n]['force_field_type'] = SMG.nodes[n]['element_symbol'] + \
                    '_' + ID_string

        # add COM sites where relevant, extend this list as new types are added
        if ID_string in ('O2', 'N2'):

            coords = []
            anchor = SMG.nodes[comp[0]]['fractional_position']

            for n in comp:
                data = SMG.nodes[n]
                data['mol_flag'] = str(mol_flag)
                fcoord = data['fractional_position']
                mic = PBC3DF_sym(fcoord, anchor)
                fcoord += mic[1]
                ccoord = np.dot(unit_cell, fcoord)
                coords.append(ccoord)

            ccom = np.average(coords, axis=0)
            fcom = np.dot(inv_unit_cell, ccom)
            index += 1

            if ID_string == 'O2':
                fft = 'O_com'
            elif ID_string == 'N2':
                fft = 'N_com'

            ndata = {'element_symbol': 'NA',
                     'mol_flag': mol_flag,
                     'index': index,
                     'force_field_type': fft,
                     'cartesian_position': ccom,
                     'fractional_position': fcom,
                     'charge': 0.0,
                     'replication': np.array([0.0,
                                              0.0,
                                              0.0]),
                     'duplicated_version_of': None}
            edata = {'sym_code': None, 'length': None, 'bond_type': None}

            add_nodes.append([index, ndata])
            add_edges.extend(
                [(index, comp[0], edata), (index, comp[1], edata)])

    for n, data in add_nodes:
        SMG.add_node(n, **data)
    for e0, e1, data in add_edges:
        SMG.add_edge(e0, e1, **data)

    ntypes = len([FF.atom_types[ty] for ty in FF.atom_types])
    # maxatomtype_wsm = len([FF.atom_types[ty] for ty in FF.atom_types])

    # maxbondtype_wsm = len([bty for bty in FF.bond_data['params']])
    # maxangletype_wsm = len([aty for aty in FF.angle_data['params']])

    nbonds = len([i for i in FF.bond_data['params']])
    nangles = len([i for i in FF.angle_data['params']])

    # try:
    #     ndihedrals = max([i for i in FF.dihedral_data['params']])
    # except ValueError:
    #     ndihedrals = 0
    #     pass
    # try:
    #     nimpropers = max([i for i in FF.improper_data['params']])
    # except ValueError:
    #     nimpropers = 0
    #     pass
    new_bond_types = {}
    new_angle_types = {}
    # new_dihedral_types = {}
    # new_improper_types = {}

    for subG, ID_string in zip([SMG.subgraph(c).copy()
                                for c in nx.connected_components(SMG)], comps):

        constants = SM_constants[ID_string]

        # add new atom types
        for name, data in sorted(subG.nodes(data=True), key=lambda x: x[0]):

            fft = data['force_field_type']
            chg = constants['pair']['charges'][fft]
            data['charge'] = chg
            SG.add_node(name, **data)

            try:

                FF.atom_types[fft] += 0

            except KeyError:

                ntypes += 1
                FF.atom_types[fft] = ntypes
                style = constants['pair']['style']
                vdW = constants['pair']['vdW'][fft]
                FF.pair_data['params'][FF.atom_types[fft]] = (
                    style, vdW[0], vdW[1])
                FF.pair_data['comments'][FF.atom_types[fft]] = [fft, fft]
                FF.atom_masses[fft] = mass_key[data['element_symbol']]

                if 'hybrid' not in FF.pair_data['style'] and style != FF.pair_data['style']:
                    FF.pair_data['style'] = ' '.join(
                        ['hybrid', FF.pair_data['style'], style])
                elif 'hybrid' in FF.pair_data['style'] and style in FF.pair_data['style']:
                    pass
                elif 'hybrid' in FF.pair_data['style'] and style not in FF.pair_data['style']:
                    FF.pair_data['style'] += ' ' + style

        # add new bonds
        used_bonds = []
        ty = nbonds
        for e0, e1, data in subG.edges(data=True):

            bonds = constants['bonds']
            fft_i = SG.nodes[e0]['force_field_type']
            fft_j = SG.nodes[e1]['force_field_type']
            # make sure the order corresponds to that in the molecule
            # dictionary
            bond = tuple(sorted([fft_i, fft_j]))

            try:

                style = bonds[bond][0]

                if bond not in used_bonds:
                    ty = ty + 1
                    new_bond_types[bond] = ty
                    FF.bond_data['params'][ty] = list(bonds[bond])
                    FF.bond_data['comments'][ty] = list(bond)

                    used_bonds.append(bond)

                if 'hybrid' not in FF.bond_data['style'] and style != FF.bond_data['style']:
                    FF.bond_data['style'] = ' '.join(
                        ['hybrid', FF.bond_data['style'], style])
                elif 'hybrid' in FF.bond_data['style'] and style in FF.bond_data['style']:
                    pass
                elif 'hybrid' in FF.bond_data['style'] and style not in FF.bond_data['style']:
                    FF.bond_data['style'] += ' ' + style

                if ty in FF.bond_data['all_bonds']:
                    FF.bond_data['count'] = (
                        FF.bond_data['count'][0] + 1,
                        FF.bond_data['count'][1] + 1)
                    FF.bond_data['all_bonds'][ty].append((e0, e1))
                else:
                    FF.bond_data['count'] = (
                        FF.bond_data['count'][0] + 1,
                        FF.bond_data['count'][1] + 1)
                    FF.bond_data['all_bonds'][ty] = [(e0, e1)]

            except KeyError:
                pass

        # add new angles
        used_angles = []
        ty = nangles
        for name, data in subG.nodes(data=True):

            angles = constants['angles']
            nbors = list(subG.neighbors(name))

            for comb in combinations(nbors, 2):

                j = name
                i, k = comb
                fft_i = subG.nodes[i]['force_field_type']
                fft_j = subG.nodes[j]['force_field_type']
                fft_k = subG.nodes[k]['force_field_type']

                angle = sorted((fft_i, fft_k))
                angle = (angle[0], fft_j, angle[1])

                try:

                    style = angles[angle][0]
                    FF.angle_data['count'] = (
                        FF.angle_data['count'][0] + 1,
                        FF.angle_data['count'][1])

                    if angle not in used_angles:
                        ty = ty + 1
                        new_angle_types[angle] = ty
                        FF.angle_data['count'] = (
                            FF.angle_data['count'][0], FF.angle_data['count'][1] + 1)
                        FF.angle_data['params'][ty] = list(angles[angle])
                        FF.angle_data['comments'][ty] = list(angle)

                        used_angles.append(angle)

                    if 'hybrid' not in FF.angle_data['style'] and style != FF.angle_data['style']:
                        FF.angle_data['style'] = ' '.join(
                            ['hybrid', FF.angle_data['style'], style])
                    elif 'hybrid' in FF.angle_data['style'] and style in FF.angle_data['style']:
                        pass
                    elif 'hybrid' in FF.angle_data['style'] and style not in FF.angle_data['style']:
                        FF.angle_data['style'] += ' ' + style

                    if ty in FF.angle_data['all_angles']:
                        FF.angle_data['all_angles'][ty].append((i, j, k))
                    else:
                        FF.angle_data['all_angles'][ty] = [(i, j, k)]

                except KeyError:
                    pass

        # add new dihedrals

    FF.bond_data['count'] = (
        FF.bond_data['count'][0], len(
            FF.bond_data['params']))
    FF.angle_data['count'] = (
        FF.angle_data['count'][0], len(
            FF.angle_data['params']))

    if 'tip4p' in FF.pair_data['style']:

        for ty, pair in FF.pair_data['comments'].items():
            fft = pair[0]
            if fft == 'O_w':
                FF.pair_data['O_type'] = ty
            if fft == 'H_w':
                FF.pair_data['H_type'] = ty

        for ty, bond in FF.bond_data['comments'].items():
            if sorted(bond) == ['H_w', 'O_w']:
                FF.pair_data['H2O_bond_type'] = ty

        for ty, angle in FF.angle_data['comments'].items():
            if angle == ['H_w', 'O_w', 'H_w']:
                FF.pair_data['H2O_angle_type'] = ty

        if 'long' in FF.pair_data['style']:
            # only TIP4P/2005 is implemented
            FF.pair_data['M_site_dist'] = 0.1546
        elif 'cut' in FF.pair_data['style'] and ff_string == 'TIP4P_2005_cutoff':
            FF.pair_data['M_site_dist'] = 0.1546
        elif 'cut' in FF.pair_data['style'] and ff_string == 'TIP4P_cutoff':
            FF.pair_data['M_site_dist'] = 0.1500


def update_potential(potential_data, new_potential_params, potential_coeff):
    write_instyles = False
    add_styles = set([new_potential_params[ty]['style']
                      for ty in new_potential_params])
    for ABS in add_styles:
        if ABS not in potential_data['style'] and 'hybrid' in potential_data['style']:
            potential_data['style'] = potential_data['style'] + ' ' + ABS
            write_instyles = True
        if ABS not in potential_data['style'] and 'hybrid' not in potential_data['style']:
            potential_data['style'] = 'hybrid ' + \
                                      potential_data['style'] + ' ' + ABS
            write_instyles = True
        else:
            pass

    if write_instyles:
        instyles = {
            ty: ' ' +
                new_potential_params[ty]['style'] for ty in new_potential_params}
    else:
        instyles = {ty: '' for ty in new_potential_params}

    potential_data['infile_add_lines'] = []
    for ty, data in new_potential_params.items():
        strparams = ' '.join([str(p) for p in data['params']])
        potential_data['infile_add_lines'].append(
            potential_coeff +
            str(ty) +
            instyles[ty] +
            ' ' +
            strparams +
            ' ' +
            data['comments'])


def include_molecule_file(FF, maxIDs, add_molecule):
    max_atom_ty, max_bond_ty, max_angle_ty, max_dihedral_ty, max_improper_ty = maxIDs
    molname, model, N = add_molecule

    if molname in ('water', 'Water', 'H2O', 'h2o'):
        molfile, LJ_params, bond_params, angle_params, molnames, mass_dict, M_site_dist, extra_types = WMF.water(
            max_atom_ty, max_bond_ty, max_angle_ty, model=model)
        dihedral_params = None
        improper_params = None
        FF.pair_data['special_bonds'] = 'lj/coul 0.0 0.0 1.0'
        FF.pair_data['O_type'] = max_atom_ty + 1
        FF.pair_data['H_type'] = max_atom_ty + 2
        FF.pair_data['H2O_bond_type'] = max_bond_ty + 1
        FF.pair_data['H2O_angle_type'] = max_angle_ty + 1
        FF.pair_data['M_site_dist'] = M_site_dist

    add_LJ_style = LJ_params['style']
    if add_LJ_style not in FF.pair_data['style']:
        FF.pair_data['style'] = FF.pair_data['style'] + ' ' + add_LJ_style
        if 'hybrid' not in FF.pair_data['style']:
            FF.pair_data['style'] = 'hybrid ' + FF.pair_data['style']

    for aty, param in LJ_params.items():
        if aty not in ('style', 'comments'):
            FF.pair_data['params'][aty] = param
            FF.pair_data['comments'][aty] = LJ_params['comments'][aty]

    if bond_params is not None:
        update_potential(FF.bond_data, bond_params, 'bond_coeff      ')
    if angle_params is not None:
        update_potential(FF.angle_data, angle_params, 'angle_coeff     ')
    if dihedral_params is not None:
        update_potential(
            FF.dihedral_params,
            dihedral_params,
            'dihedral_coeff  ')
    if improper_params is not None:
        update_potential(FF.improper_data, improper_params, 'improper_coeff  ')

    infile_add_lines = ['molecule        ' + ' '.join(molnames)]
    for atom in mass_dict:
        infile_add_lines.append('mass            ' +
                                str(atom) + ' ' + str(mass_dict[atom]))
    seed0 = randint(1, 10000)
    seed1 = randint(1, 10000)

    if N > 0:
        create_line = ' '.join(
            [str(N), str(seed0), 'NULL', 'mol', molnames[0], str(seed1), 'units', 'box'])
        infile_add_lines.append('create_atoms    0 random ' + create_line)

    return molfile, infile_add_lines, extra_types


def read_RASPA_pdb(file):
    with open(file, 'r') as pdb:
        pdb = pdb.read()
        pdb = pdb.split('\n')

    atoms = Atoms()
    for line in pdb:
        s = line.split()
        if len(s) > 0 and s[0] == 'ATOM':
            atoms.append(Atom(s[2], np.array([float(c) for c in s[4:7]])))

    return atoms


def read_small_molecule_file(sm_file, system):
    fm = sm_file.split('.')[-1]
    max_ind = max(system['graph'])
    ind = max_ind + 1

    if fm not in ('pdb', 'xyz', 'cif'):
        raise ValueError(
            'only xyz and RASPA pdb formats are supported for small molecule files')

    if fm == 'pdb':
        logger.debug('assuming small molecule file is in RASPA pdb format, if not, too bad...')
        atoms = read_RASPA_pdb(sm_file)
    else:
        atoms = read(sm_file, format=fm)

    atoms.set_cell(system['box'])
    SMG = nx.Graph()

    for atom in atoms:
        SMG.add_node(ind,
                     element_symbol=atom.symbol,
                     mol_flag='1',
                     index=ind,
                     force_field_type='',
                     cartesian_position=atom.position,
                     fractional_position=atom.scaled_position,
                     charge=0.0,
                     replication=np.array([0.0,
                                           0.0,
                                           0.0]),
                     duplicated_version_of=None)

        ind += 1

    # don't want to overwrite extra framework species already in the cif
    system['SM_graph'] = nx.compose(SMG, system['SM_graph'])
