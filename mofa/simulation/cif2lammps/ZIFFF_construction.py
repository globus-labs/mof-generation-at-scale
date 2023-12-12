import logging
import itertools
import warnings
from . import atomic_data
import networkx as nx
import numpy as np
from .cif2system import PBC3DF_sym
from .pymatgen_cif2system import M
from .force_field_construction import force_field
from collections import Counter
from . import gaff
from . import gaff2
from . import ZIFFF_constants

metals = atomic_data.metals
mass_key = atomic_data.mass_key

logger = logging.getLogger(__name__)


def parameter_loop(options, type_dict):
    params = None
    for option in options:

        try:
            params = type_dict[option]
            break
        except KeyError:
            continue

    if params is not None:
        return params
    else:
        raise (ValueError('No parameters found for', options))


def dihedral_parameter_loop(options, type_dict):
    params = None
    for option in options:

        try:
            params = type_dict[option]
            break
        except KeyError:
            continue

    if params is not None:
        return params
    else:
        message = 'No parameters found for ' + ' '.join(option)
        warnings.warn(message)
        return None


def read_gaffdat(mode='gaff'):
    if mode == 'gaff':

        gaff_atom_types = gaff.gaff_atom_types
        gaff_LJ_parameters = gaff.gaff_LJ_params
        gaff_bonds = gaff.gaff_bonds
        gaff_angles = gaff.gaff_angles
        gaff_dihedrals = gaff.gaff_dihedrals
        gaff_impropers = gaff.gaff_impropers

    elif mode == 'gaff2':

        gaff_atom_types = gaff2.gaff_atom_types
        gaff_LJ_parameters = gaff2.gaff_LJ_params
        gaff_bonds = gaff2.gaff_bonds
        gaff_angles = gaff2.gaff_angles
        gaff_dihedrals = gaff2.gaff_dihedrals
        gaff_impropers = gaff2.gaff_impropers

    else:
        raise ValueError('mode must be gaff or gaff2')

    gaff_atom_types = [
        LL.split() for LL in gaff_atom_types.split('\n') if len(
            LL.split()) > 0]
    gaff_atom_types = dict((LL[0], (float(LL[1]), float(LL[2])))
                           for LL in gaff_atom_types)

    gaff_LJ_parameters = [
        LL.split() for LL in gaff_LJ_parameters.split('\n') if len(
            LL.split()) > 0]
    gaff_LJ_parameters = dict(
        (LL[0], (float(LL[2]), float(LL[1]))) for LL in gaff_LJ_parameters)

    gaff_bond_list = [(''.join(LL[0:5].split()), ''.join(LL[5:16].split()), ''.join(
        LL[16:26].split())) for LL in gaff_bonds.split('\n') if len(LL.split()) > 0]
    gaff_bonds = {}
    for LL in gaff_bond_list:
        bond, K, r0 = LL
        bond = tuple(sorted(bond.split('-')))
        gaff_bonds[bond] = (float(K), float(r0))

    gaff_angle_list = [(''.join(LL[0:8].split()), ''.join(LL[8:20].split()), ''.join(
        LL[20:30].split())) for LL in gaff_angles.split('\n') if len(LL.split()) > 0]
    gaff_angles = {}
    for LL in gaff_angle_list:
        angle, K, theta0 = LL
        angle = tuple(angle.split('-'))
        gaff_angles[angle] = (float(K), float(r0))

    gaff_dihedral_list = [(''.join(LL[0:11].split()), ''.join(LL[11:19].split()), ''.join(LL[19:31].split()), ''.join(
        LL[31:48].split()), ''.join(LL[48:52].split())) for LL in gaff_dihedrals.split('\n') if len(LL.split()) > 0]
    gaff_dihedrals = {}
    for LL in gaff_dihedral_list:
        dihedral, m, K, psi0, n = LL
        dihedral = tuple(dihedral.split('-'))
        try:
            gaff_dihedrals[dihedral].extend(
                [float(K) / int(float(m)), int(float(n)), float(psi0)])
        except KeyError:
            gaff_dihedrals[dihedral] = [
                float(K) / int(float(m)), int(float(n)), float(psi0)]

    gaff_improper_list = [(''.join(LL[0:11].split()), ''.join(LL[11:33].split()), ''.join(LL[33:47].split()),
                           ''.join(LL[47:50].split())) for LL in gaff_impropers.split('\n') if len(LL.split()) > 0]
    gaff_impropers = {}
    for LL in gaff_improper_list:
        improper, K, psi0, n = LL
        improper = tuple(improper.split('-'))
        gaff_impropers[improper] = (float(K), int(float(n)), float(psi0))

    gaff_data = {
        'types': gaff_atom_types,
        'bonds': gaff_bonds,
        'angles': gaff_angles,
        'dihedrals': gaff_dihedrals,
        'impropers': gaff_impropers,
        'LJ_parameters': gaff_LJ_parameters}
    return gaff_data


def GAFF_type(atom, data, SG, pure_aromatic_atoms, aromatic_atoms):
    """
        returns GAFF atom types, this can be expanded to the full 97 types
        apply to atoms in this order:
        (1) nitrogens, carbons, halogens
]       (2) hydrogens
    """

    sym = data['element_symbol']
    nbors = [a for a in SG.neighbors(atom)]
    nbor_symbols = [SG.nodes[n]['element_symbol'] for n in nbors]
    bond_types = [SG.get_edge_data(atom, n)['bond_type'] for n in nbors]

    doubleO = False
    doubleS = False

    for elem, bt in zip(nbor_symbols, bond_types):
        if elem == 'O' and bt == 'D':
            doubleO = True
        elif elem == 'S' and bt == 'D':
            doubleS = True

    ty = None
    hyb = None

    if sym == 'C':
        # sp1 carbons
        if len(nbors) == 2:
            ty = 'c1'
            hyb = 'sp1'
        # sp2 carbons
        elif len(nbors) == 3:
            # aromatic carbons
            if 'A' in bond_types:
                # pure aromatic C are in benzene/pyridine
                # (https://github.com/rsdefever/GAFF-foyer)
                if atom in pure_aromatic_atoms:
                    ty = 'ca'
                # non-pure aromatic C (anything else with aromatic bond)
                else:
                    ty = 'cc'
            elif 'D' in bond_types:
                # R-(C=O)-R
                if doubleO and not doubleS:
                    ty = 'c'
                # R-(C=S)-R
                elif doubleS and not doubleO:
                    ty = 'cs'
                # non-aromatic sp2 carbon
                elif not doubleO and not doubleS:
                    ty = 'c2'
                # other cases not considered
                else:
                    ty = None
            hyb = 'sp2'
        # sp3 carbons
        elif len(nbors) == 4:
            ty = 'c3'
            hyb = 'sp3'

        # note that cp, cq, cd, ce, cf, cg, ch, cx, cy, cu, cv, cz are are
        # probably not needed for ZIF-FF

    elif sym == 'H':

        # make sure to apply this function to hydrogens after all other atoms
        # have been typed
        if len(nbors) != 1:
            raise ValueError('H with more than one neighbor found')

        nbor = nbors[0]
        nbor_sym = nbor_symbols[0]
        nbor_hyb = SG.nodes[nbor]['hybridization']

        if nbor_sym == 'C':
            # bonded to aromatic atom
            if nbor in aromatic_atoms:
                ty = 'ha'
                hyb = 'sp1'
            else:
                # bonded to sp3 carbon, need to add electron withdrawing cases
                if nbor_hyb == 'sp3':
                    ty = 'h1'
                    hyb = 'sp1'
                # bonded to non-aromatic sp2 carbon, need to add electron
                # withdrawing cases
                elif nbor_hyb == 'sp2':
                    ty = 'h4'
                    hyb = 'sp1'
        # bonded to N
        elif nbor_sym == 'N':
            ty = 'hn'
            hyb = 'sp1'
        # bonded to O
        elif nbor_sym == 'O':
            ty = 'ho'
            hyb = 'sp1'
        else:
            ty = None
            hyb = None

    elif sym == 'F':
        # halogens are easy, only one type of each
        ty = 'f'
        hyb = 'sp1'
    elif sym == 'Cl':
        # halogens are easy, only one type of each
        ty = 'cl'
        hyb = 'sp1'
    elif sym == 'Br':
        # halogens are easy, only one type of each
        ty = 'br'
        hyb = 'sp1'
    elif sym == 'I':
        # halogens are easy, only one type of each
        ty = 'i'
        hyb = 'sp1'

    elif sym == 'N':
        if len(nbors) == 1:
            ty = 'n'
            hyb = 'sp1'
        elif len(nbors) == 2:
            if 'A' in bond_types:
                if atom in pure_aromatic_atoms:
                    ty = 'nb'
                else:
                    ty = 'nc'
            else:
                ty = 'n2'
            hyb = 'sp2'
        elif len(nbors) == 3:
            # aromatic nitrogens
            if 'A' in bond_types and Counter(nbor_symbols)['H'] <= 1:
                # N in pyridine
                if atom in pure_aromatic_atoms:
                    ty = 'nb'
                # other N with aromatic bonds
                else:
                    ty = 'nc'
            # -NH2 bound to an aromatic atom, important for functionalized ZIFs
            elif Counter(nbor_symbols)['H'] == 2 and any(n in aromatic_atoms for n in nbors):
                ty = 'nh'
            # other sp3 N with 3 neighbors
            else:
                ty = 'n3'
            hyb = 'sp3'
        elif len(nbors) == 4:
            # sp3 N with 4 neighbors
            ty = 'n4'
            hyb = 'sp3'

        # note that nd-n6 (see gaff2.dat) are probably not needed for ZIF-FF

    elif sym == 'O':
        # O with one neighbor
        if len(nbors) == 1:
            ty = 'o'
            hyb = 'sp1'
        if len(nbors) == 2:
            # R-OH
            if Counter(nbor_symbols)['H'] == 1:
                ty = 'oh'
                hyb = 'sp2'
            # ether and ester O
            else:
                ty = 'os'
                hyb = 'sp2'
    # add these if needed
    elif sym == 'P':
        pass
    elif sym == 'S':
        pass

    if (ty is None or hyb is None) and sym != 'H':
        raise ValueError(
            'No GAFF atom type identified for element',
            sym,
            'with neighbors',
            nbor_symbols)

    return ty, hyb


class ZIFFF(force_field):

    def __init__(self, system, cutoff, args):

        pi = np.pi
        a, b, c, alpha, beta, gamma = system['box']
        ax = a
        ay = 0.0
        az = 0.0
        bx = b * np.cos(gamma * pi / 180.0)
        by = b * np.sin(gamma * pi / 180.0)
        bz = 0.0
        cx = c * np.cos(beta * pi / 180.0)
        cy = (c * b * np.cos(alpha * pi / 180.0) - bx * cx) / by
        cz = (c ** 2.0 - cx ** 2.0 - cy ** 2.0) ** 0.5
        self.unit_cell = np.asarray(
            [[ax, ay, az], [bx, by, bz], [cx, cy, cz]]).T

        NMG = system['graph'].copy()
        edge_list = list(NMG.edges())

        for e0, e1 in edge_list:

            sym0 = NMG.nodes[e0]['element_symbol']
            sym1 = NMG.nodes[e1]['element_symbol']

            if sym0 in metals or sym1 in metals:
                NMG.remove_edge(e0, e1)

        linkers = nx.connected_components(NMG)
        pure_aromatic_atoms = []
        aromatic_atoms = []

        for linker in linkers:

            SG = NMG.subgraph(linker)
            all_cycles = nx.simple_cycles(nx.to_directed(SG))
            all_cycles = set([tuple(sorted(cy))
                              for cy in all_cycles if len(cy) > 4])

            for cycle in all_cycles:

                # rotate the ring normal vec onto the z-axis to determine
                # coplanarity
                fcoords = np.array(
                    [system['graph'].nodes[c]['fractional_position'] for c in cycle])
                element_symbols = [system['graph'].nodes[c]
                                   ['element_symbol'] for c in cycle]
                anchor = fcoords[0]
                fcoords = np.array(
                    [vec - PBC3DF_sym(anchor, vec)[1] for vec in fcoords])
                coords = np.dot(self.unit_cell.T, fcoords.T).T

                coords -= np.average(coords, axis=0)

                vec0 = coords[0]
                vec1 = coords[1]

                normal = np.cross(vec0, vec1)
                RZ = M(normal, np.array([0.0, 0.0, 1.0]))
                coords = np.dot(RZ, coords.T).T
                maxZ = max([abs(z) for z in coords[:, -1]])

                # if coplanar make all bond orders 1.5
                if maxZ < 0.1:
                    aromatic_atoms.extend(list(cycle))
                    if Counter(element_symbols)['C'] == 6:
                        pure_aromatic_atoms.extend(list(cycle))
                    elif Counter(element_symbols)['C'] == 5 and Counter(element_symbols)['N'] == 1:
                        pure_aromatic_atoms.extend(list(cycle))

        self.pure_aromatic_atoms = pure_aromatic_atoms
        self.aromatic_atoms = aromatic_atoms
        args['FF_parameters'] = read_gaffdat(mode='gaff')
        self.system = system
        self.cutoff = cutoff
        self.args = args
        self.ZIFFF_types = ('C1', 'C2', 'C3', 'N', 'Zn', 'H2', 'H3')

    def type_atoms(self):

        SG = self.system['graph']
        types = []
        imidazolate_ring_atoms = []

        for atom, data in SG.nodes(data=True):

            element_symbol = data['element_symbol']

            if element_symbol in ('Zn', 'Cd'):
                nborhood = list(nx.ego_graph(SG, atom, radius=2))
                imidazolate_ring_atoms.extend(nborhood)

        # assign Zn, N, and imidazolate ring types first
        imidazolate_gaff_types = {}
        for atom in SG.nodes(data=True):

            hyb = None
            name, inf = atom
            element_symbol = inf['element_symbol']
            nbors = [a for a in SG.neighbors(name)]
            nbor_symbols = [SG.nodes[n]['element_symbol'] for n in nbors]
            mass = mass_key[element_symbol]

            # one type of Zn
            if element_symbol == 'Zn':
                ty = 'Zn'
                hyb = None
                types.append((ty, element_symbol, mass))
                SG.node[name]['force_field_type'] = ty
                SG.node[name]['hybridization'] = hyb

            # imidazolate ring atoms
            if name in imidazolate_ring_atoms:
                # one type of N
                if element_symbol == 'N':
                    ty = 'N'
                    hyb = 'sp2'
                # three types of C
                if element_symbol == 'C':
                    if Counter(nbor_symbols)['N'] == 2:
                        ty = 'C1'
                        hyb = 'sp2'
                    elif Counter(nbor_symbols)['N'] == 1:
                        ty = 'C2'
                        hyb = 'sp2'

                types.append((ty, element_symbol, mass))
                SG.node[name]['force_field_type'] = ty
                SG.node[name]['hybridization'] = hyb

                if element_symbol not in metals:
                    gaff_ty, gaff_hyb = GAFF_type(
                        name, inf, SG, self.pure_aromatic_atoms, self.aromatic_atoms)
                    imidazolate_gaff_types[ty] = gaff_ty

        # type non-imidazolate-ring atoms
        for atom in SG.nodes(data=True):

            name, inf = atom

            if inf['force_field_type'] == '':

                element_symbol = inf['element_symbol']
                nbors = [a for a in SG.neighbors(name)]
                nbor_symbols = [SG.nodes[n]['element_symbol'] for n in nbors]
                nbor_types = [SG.nodes[n]['force_field_type'] for n in nbors]
                mass = mass_key[element_symbol]

                # imidazolate ring atom adjacent that are not hydrogens
                if element_symbol != 'H':
                    if name not in imidazolate_ring_atoms and any(
                            n in imidazolate_ring_atoms for n in nbors):
                        # special case for ZIF-8 -CH3 group which has an
                        # explicit type in ZIF-FF
                        if element_symbol == 'C' and sorted(nbor_symbols) == ['C', 'H', 'H', 'H'] and 'C1' in nbor_symbols:
                            ty = 'C1'
                            hyb = 'sp3'
                        # other functionalizations are typed for GAFF
                        else:
                            ty, hyb = GAFF_type(
                                name, inf, SG, self.pure_aromatic_atoms, self.aromatic_atoms)
                    elif name not in imidazolate_ring_atoms and not any(n in imidazolate_ring_atoms for n in nbors):
                        ty, hyb = GAFF_type(
                            name, inf, SG, self.pure_aromatic_atoms, self.aromatic_atoms)

                types.append((ty, element_symbol, mass))
                SG.node[name]['force_field_type'] = ty
                SG.node[name]['hybridization'] = hyb

        # type hydrogens last
        for atom in SG.nodes(data=True):

            name, inf = atom
            element_symbol = inf['element_symbol']
            nbors = [a for a in SG.neighbors(name)]
            nbor_symbols = [SG.nodes[n]['element_symbol'] for n in nbors]
            nbor_types = [SG.nodes[n]['force_field_type'] for n in nbors]
            mass = mass_key[element_symbol]

            # imidazolate ring atom adjacent that are not hydrogens
            if element_symbol == 'H':

                if 'C2' in nbor_types:
                    ty = 'H2'
                    hyb = 'sp1'
                elif 'C3' in nbor_types:
                    ty = 'H3'
                    hyb = 'sp1'
                else:
                    ty, hyb = GAFF_type(
                        name, inf, SG, self.pure_aromatic_atoms, self.aromatic_atoms)

                types.append((ty, element_symbol, mass))
                SG.node[name]['force_field_type'] = ty
                SG.node[name]['hybridization'] = hyb

        types = set(types)
        Ntypes = len(types)
        atom_types = dict((ty[0], i + 1)
                          for i, ty in zip(range(Ntypes), types))
        atom_element_symbols = dict((ty[0], ty[1]) for ty in types)
        atom_masses = dict((ty[0], ty[2]) for ty in types)

        # for n, data in SG.nodes(data=True):
        #
        # sym = data['element_symbol']
        # nbors = [a for a in SG.neighbors(name)]
        # nbor_symbols = [SG.nodes[n]['element_symbol'] for n in nbors]
        #
        # if n in self.pure_aromatic_atoms:
        # logger.debug(sym, nbor_symbols, data['force_field_type'])

        self.system['graph'] = SG
        self.atom_types = atom_types
        self.atom_element_symbols = atom_element_symbols
        self.atom_masses = atom_masses
        self.imidazolate_gaff_types = imidazolate_gaff_types

    def bond_parameters(self, bond):

        gaff_bonds = self.args['FF_parameters']['bonds']
        i, j = sorted(bond)

        if all(t in self.ZIFFF_types for t in (i, j)):

            k_ij, r_ij = parameter_loop(
                [bond, bond[::-1]], ZIFFF_constants.ZIFFF_bonds)
            params = ('harmonic', k_ij, r_ij)

        else:

            new_bond = tuple(sorted(
                [self.imidazolate_gaff_types[ty] if ty in self.imidazolate_gaff_types else ty for ty in bond]))
            k_ij, r_ij = parameter_loop(
                [bond, bond[::-1], new_bond, new_bond[::-1]], gaff_bonds)
            params = ('harmonic', k_ij, r_ij)

        return params

    def angle_parameters(self, angle):

        gaff_angles = self.args['FF_parameters']['angles']
        i, j, k = angle

        if all(t in self.ZIFFF_types for t in (i, j, k)):

            K, theta0 = parameter_loop(
                [angle, angle[::-1]], ZIFFF_constants.ZIFFF_angles)
            params = ('harmonic', K, theta0)

        else:

            new_angle = tuple([self.imidazolate_gaff_types[ty]
                               if ty in self.imidazolate_gaff_types else ty for ty in angle])
            K, theta0 = parameter_loop(
                [angle, angle[::-1], new_angle, new_angle[::-1]], gaff_angles)
            if K is None:
                logger.debug(new_angle)
            params = ('harmonic', K, theta0)

        return params

    def dihedral_parameters(self, dihedral):

        gaff_dihedrals = self.args['FF_parameters']['dihedrals']
        i, j, k, LL = dihedral

        if all(t in self.ZIFFF_types for t in (i, j, k, LL)):

            params = dihedral_parameter_loop(
                [dihedral, dihedral[::-1]], ZIFFF_constants.ZIFFF_dihedrals)

            if params is not None:
                K, n, d = params
                params = ('fourier', 1, K, n, d)

        else:

            X_dihedral = ('X', dihedral[1], dihedral[2], 'X')
            new_X_dihedral = tuple(
                [self.imidazolate_gaff_types[ty] if ty in self.imidazolate_gaff_types else ty for ty in X_dihedral])
            new_dihedral = tuple([self.imidazolate_gaff_types[ty]
                                  if ty in self.imidazolate_gaff_types else ty for ty in dihedral])

            options = [X_dihedral,
                       X_dihedral[::-1],
                       new_X_dihedral,
                       new_X_dihedral[::-1],
                       dihedral,
                       dihedral[::-1],
                       new_dihedral,
                       new_dihedral[::-1]]
            params = dihedral_parameter_loop(options, gaff_dihedrals)

            if params is not None:
                Nterms = len(params) % 3
                params = tuple(['fourier', Nterms] + params)

        return params

    def improper_parameters(self, improper):

        gaff_impropers = self.args['FF_parameters']['impropers']
        i, j, k, LL = improper
        params = None

        if all(t in self.ZIFFF_types for t in (i, j, k, LL)):

            # the ZIF-FF paper lists the central atom first
            improper_combs = [(i, j, k, LL),
                              (i, j, LL, k),
                              (i, k, j, LL),
                              (i, k, LL, j),
                              (i, LL, j, k),
                              (i, LL, j, k)]

            for imp in improper_combs:
                try:
                    K, n, d = ZIFFF_constants.ZIFFF_impropers[imp]
                    params = ('cvff', K, -1, 2)
                    break
                except KeyError:
                    continue

        else:

            improper_combs = [(j, k, i, LL),
                              (j, LL, i, k),
                              (k, j, i, LL),
                              (k, LL, i, j),
                              (LL, j, i, k),
                              (LL, j, i, k)]

            for imp in improper_combs:
                try:
                    K, psi0, n = gaff_impropers[imp]
                    params = ('cvff', K, -1, 2)
                    break
                except KeyError:
                    continue

            if params is None:

                i, j, k, LL = tuple(
                    [self.imidazolate_gaff_types[ty] if ty in self.imidazolate_gaff_types else ty for ty in improper])
                improper_combs = [(j, k, i, LL),
                                  (j, LL, i, k),
                                  (k, j, i, LL),
                                  (k, LL, i, j),
                                  (LL, j, i, k),
                                  (LL, j, i, k)]

                if i in ('c', 'ca', 'n', 'n2', 'na'):
                    ximps = [('X', k, i, LL),
                             ('X', LL, i, k),
                             ('X', j, i, LL),
                             ('X', LL, i, j),
                             ('X', j, i, k),
                             ('X', j, i, k),
                             ('X', 'X', i, LL),
                             ('X', 'X', i, k),
                             ('X', 'X', i, LL),
                             ('X', 'X', i, j),
                             ('X', 'X', i, k),
                             ('X', 'X', i, k)]

                    improper_combs.extend(ximps)

                for imp in improper_combs:
                    try:
                        K, psi0, n = gaff_impropers[imp]
                        params = ('cvff', K, -1, 2)
                        break
                    except KeyError:
                        continue

        if params is None:
            message = 'No improper type identified for ' + ' '.join(list(imp))
            warnings.warn(message)

        return params

    def pair_parameters(self, charges=False):

        gaff_LJ_parameters = self.args['FF_parameters']['LJ_parameters']
        atom_types = self.atom_types
        all_params = {}
        comments = {}

        # determine style and special bonds
        if charges:
            style = 'lj/cut/coul/long'
            sb = 'lj 0.0 0.0 0.5 coul 0.0 0.0 0.6874'
        else:
            warnings.warn(
                'ZIF-FF or any AMBER based force-field always uses charges, your results will be incorrect without them')
            style = 'lj/cut'
            sb = 'lj 0.0 0.0 1.0'

        for a in atom_types:

            if a in self.ZIFFF_types:
                eps, sig, charge = ZIFFF_constants.ZIFFF_LJ_parameters[a]
                params = (eps, sig)
                all_params[a] = params

            else:
                eps, rmin = gaff_LJ_parameters[a]
                params = (eps, 2 * rmin * (2 ** (-1.0 / 6.0)))
                all_params[a] = params

            comments[a] = [a, a]

        self.pair_data = {
            'params': all_params,
            'style': style,
            'special_bonds': sb,
            'comments': comments}

    def enumerate_bonds(self):

        SG = self.system['graph']

        bonds = {}
        for e in SG.edges(data=True):

            i, j, data = e
            fft_i = SG.node[i]['force_field_type']
            fft_j = SG.node[j]['force_field_type']
            bond = tuple(sorted([fft_i, fft_j]))

            # add to list if bond type already exists, else add a new type
            try:
                bonds[bond].append((i, j))
            except KeyError:
                bonds[bond] = [(i, j)]

        bond_params = {}
        bond_comments = {}
        all_bonds = {}
        ID = 0
        count = 0
        # index bonds by ID
        for b in bonds:
            ID += 1
            bond = (b[0], b[1])
            params = self.bond_parameters(bond)
            bond_params[ID] = list(params)
            bond_comments[ID] = list(bond)
            all_bonds[ID] = bonds[b]
            count += len(bonds[b])

        self.bond_data = {
            'all_bonds': all_bonds,
            'params': bond_params,
            'style': 'harmonic',
            'count': (
                count,
                len(all_bonds)),
            'comments': bond_comments}

    def enumerate_angles(self):

        SG = self.system['graph']
        angles = {}

        for n in SG.nodes(data=True):

            name, data = n
            nbors = list(SG.neighbors(name))

            for comb in itertools.combinations(nbors, 2):

                j = name
                i, k = comb

                fft_i = SG.node[i]['force_field_type']
                fft_j = SG.node[j]['force_field_type']
                fft_k = SG.node[k]['force_field_type']

                sort_ik = sorted([(fft_i, i), (fft_k, k)], key=lambda x: x[0])
                fft_i, i = sort_ik[0]
                fft_k, k = sort_ik[1]

                angle = sorted((fft_i, fft_k))
                angle = (angle[0], fft_j, angle[1])

                # add to list if angle type already exists, else add a new type
                try:
                    angles[angle].append((i, j, k))
                except KeyError:
                    angles[angle] = [(i, j, k)]

        angle_params = {}
        angle_comments = {}
        all_angles = {}
        ID = 0
        count = 0
        styles = []

        # index angles by ID
        for a in angles:
            ID += 1
            fft_i, fft_j, fft_k = a
            angle = (fft_i, fft_j, fft_k)
            params = self.angle_parameters(angle)
            styles.append(params[0])
            angle_params[ID] = list(params)
            angle_comments[ID] = list(angle)
            all_angles[ID] = angles[a]
            count += len(angles[a])

        styles = set(styles)
        if len(styles) == 1:
            style = list(styles)[0]
        else:
            style = 'hybrid ' + ' '.join(styles)

        self.angle_data = {
            'all_angles': all_angles,
            'params': angle_params,
            'style': style,
            'count': (
                count,
                len(all_angles)),
            'comments': angle_comments}

    def enumerate_dihedrals(self):

        SG = self.system['graph']
        dihedrals = {}
        dihedral_params = {}

        for e in SG.edges(data=True):

            j = e[0]
            k = e[1]
            fft_j = SG.node[j]['force_field_type']
            fft_k = SG.node[k]['force_field_type']

            nbors_j = [n for n in SG.neighbors(j) if n != k]
            nbors_k = [n for n in SG.neighbors(k) if n != j]

            il_pairs = list(itertools.product(nbors_j, nbors_k))
            dihedral_list = [(SG.node[p[0]]['force_field_type'], fft_j,
                              fft_k, SG.node[p[1]]['force_field_type']) for p in il_pairs]

            for dihedral in dihedral_list:

                params = self.dihedral_parameters(dihedral)

                if params is not None:

                    try:
                        dihedrals[dihedral].append(dihedral)
                    except KeyError:
                        dihedrals[dihedral] = [dihedral]
                        dihedral_params[dihedral] = params

        all_dihedrals = {}
        dihedral_comments = {}
        indexed_dihedral_params = {}
        ID = 0
        count = 0

        for d in dihedrals:
            ID += 1
            params = dihedral_params[d]
            all_dihedrals[ID] = dihedrals[d]
            indexed_dihedral_params[ID] = list(dihedral_params[d])
            dihedral_comments[ID] = list(d)
            count += len(dihedrals[d])

        self.dihedral_data = {
            'all_dihedrals': all_dihedrals,
            'params': indexed_dihedral_params,
            'style': 'harmonic',
            'count': (
                count,
                len(all_dihedrals)),
            'comments': dihedral_comments}

    def enumerate_impropers(self):

        SG = self.system['graph']
        impropers = {}

        for n in SG.nodes(data=True):

            i, data = n
            nbors = list(SG.neighbors(i))

            if len(nbors) == 3:

                nbors = sorted([(m, SG.node[m]['force_field_type'])
                                for m in nbors], key=lambda x: x[1])
                fft_nbors = [m[1] for m in nbors]
                nbors = [m[0] for m in nbors]

                fft_i = data['force_field_type']
                j, k, LL = nbors
                imp = [i, j, k, LL]
                fft_j, fft_k, fft_l = fft_nbors
                imp_type = (fft_i, fft_j, fft_k, fft_l)

                try:
                    impropers[imp_type].append(imp)
                except KeyError:
                    impropers[imp_type] = [imp]

        all_impropers = {}
        improper_params = {}
        improper_comments = {}
        ID = 0
        count = 0
        for i in impropers:

            params = self.improper_parameters(i)

            if params is not None:
                ID += 1
                improper_params[ID] = list(params)
                improper_comments[ID] = list(i)
                all_impropers[ID] = impropers[i]
                count += len(impropers[i])

        self.improper_data = {
            'all_impropers': all_impropers,
            'params': improper_params,
            'style': 'fourier',
            'count': (
                count,
                len(all_impropers)),
            'comments': improper_comments}

    def compile_force_field(self, charges=False):

        self.type_atoms()
        self.pair_parameters(charges)
        self.enumerate_bonds()
        self.enumerate_angles()
        self.enumerate_dihedrals()
        self.enumerate_impropers()
