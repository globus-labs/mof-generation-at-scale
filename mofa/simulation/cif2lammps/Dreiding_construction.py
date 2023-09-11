from __future__ import print_function
import numpy as np
import math
import itertools
from . import atomic_data
from .force_field_construction import force_field

metals = atomic_data.metals
mass_key = atomic_data.mass_key


class Dreiding(force_field):

    def __init__(self, system, cutoff, args):

        self.system = system
        self.cutoff = cutoff
        self.args = args

    def type_atoms(self):

        SG = self.system['graph']
        types = []

        for atom in SG.nodes(data=True):

            name, inf = atom
            element_symbol = inf['element_symbol']
            nbors = list(SG.neighbors(name))
            nbor_symbols = [SG.nodes[n]['element_symbol'] for n in nbors]
            bond_types = [
                SG.get_edge_data(
                    name,
                    n)['bond_type'] for n in nbors]
            mass = mass_key[element_symbol]

            # Atom typing for Dreiding, this can be made much more robust with pattern matching,
            # but this works for most ToBaCCo MOFs, use at your own risk.
            ty = None
            if 'A' in bond_types and element_symbol != 'O':
                ty = element_symbol + '_' + 'R'
                hyb = 'resonant'
            else:
                # Group 1
                if element_symbol == 'H':
                    ty = element_symbol + '_'
                    hyb = 'sp1'
                # Group 6
                elif element_symbol in ('C', 'Si'):
                    ty = element_symbol + '_' + str(len(nbors) - 1)
                    hyb = 'sp' + str(len(nbors) - 1)
                # Group 7
                elif element_symbol in ('N'):
                    ty = element_symbol + '_' + str(len(nbors))
                    hyb = 'sp' + str(len(nbors))
                # Group 8
                elif element_symbol in ('O', 'S'):
                    # oxygen case is complex with the UFF4MOF oxygen types
                    if element_symbol == 'O':
                        # -OH, for example
                        if len(nbors) == 2 and 'A' not in bond_types and 'D' not in bond_types and not any(
                                i in metals for i in nbor_symbols):
                            ty = 'O_3'
                            hyb = 'sp3'
                        # furan oxygen, for example
                        elif len(nbors) == 2 and 'A' in bond_types and not any(i in metals for i in nbor_symbols):
                            ty = 'O_R'
                            hyb = 'sp2'
                        # carboxyllic oxygen
                        elif len(nbors) == 2 and 'D' in bond_types and not any(i in metals for i in nbor_symbols):
                            ty = 'O_2'
                            hyb = 'sp2'
                        # carboxylate oxygen bound to metal node
                        elif len(nbors) == 2 and any(i in metals for i in nbor_symbols):
                            ty = 'O_2_M'
                            hyb = 'sp2'
                        # central 3-connected oxygen
                        elif len(nbors) == 3 and all(i in metals for i in nbor_symbols):
                            ty = 'O_2_M'
                            hyb = 'sp2'
                        # node oxygens bound to metals
                        elif len(nbors) >= 3 and any(i in metals for i in nbor_symbols):
                            ty = 'O_3_M'
                            hyb = 'sp2'
                        else:
                            raise ValueError(
                                'Oxygen with neighbors ' +
                                ' '.join(nbor_symbols) +
                                ' is not parametrized')
                    # sulfur case is simple
                    elif element_symbol == 'S':
                        ty = 'S_' + str(len(nbors) + 1)
                        hyb = 'sp' + str(len(nbors) + 1)
                # Group 9
                elif element_symbol in ('F', 'Br'):
                    ty = element_symbol + '_'
                    hyb = 'sp1'
                # Metals
                elif element_symbol in metals:
                    ty = element_symbol
                    hyb = 'NA'
                # if no type can be identified
                else:
                    raise ValueError(
                        'No Dreiding type identified for ' +
                        element_symbol +
                        'with neighbors ' +
                        ' '.join(nbor_symbols))

            types.append((ty, element_symbol, mass))
            SG.nodes[name]['force_field_type'] = ty
            SG.nodes[name]['hybridization'] = hyb

        types = set(types)
        Ntypes = len(types)
        atom_types = dict((ty[0], i + 1)
                          for i, ty in zip(range(Ntypes), types))
        atom_element_symbols = dict((ty[0], ty[1]) for ty in types)
        atom_masses = dict((ty[0], ty[2]) for ty in types)

        self.system['graph'] = SG
        self.atom_types = atom_types
        self.atom_element_symbols = atom_element_symbols
        self.atom_masses = atom_masses

    def bond_parameters(self, bond, bond_order):

        Dreiding_atom_parameters = self.args['FF_parameters']

        i, j = bond
        params_i = Dreiding_atom_parameters[i]
        params_j = Dreiding_atom_parameters[j]

        R1_i, theta_i, R0_i, D0_i, phi_i, S_i = params_i
        R1_j, theta_j, R0_j, D0_j, phi_j, S_j = params_j

        r_ij = R1_i + R1_j - 0.01
        k_ij = 0.5 * bond_order * 700

        return ('harmonic', k_ij, r_ij)

    def angle_parameters(self, angle):

        Dreiding_atom_parameters = self.args['FF_parameters']

        fft_i, fft_j, fft_k = angle
        R1_j, theta_j, R0_j, D0_j, phi_j, S_j = Dreiding_atom_parameters[fft_j]

        K = 100.0

        if theta_j not in (90.0, 180.0):
            theta0 = theta_j
            angle_style = 'cosine/squared'
        elif theta_j == 180.0:
            K *= 0.5
            angle_style = 'cosine'
            return (angle_style, K)
        elif theta_j == 90.0:
            K *= 0.5
            theta0 = theta_j
            angle_style = 'cosine/periodic'
            n = 4
            b = 1
            return (angle_style, K, b, n)

        sinT0 = np.sin(math.radians(theta_j))
        K = (0.5 * K) / (sinT0 * sinT0)

        return (angle_style, K, theta0)

    def dihedral_parameters(self, bond, hybridization, element_symbols, nodes):

        fft_j, fft_k, bond_order = bond
        hyb_j, hyb_k = hybridization
        els_j, els_k = element_symbols
        node_j, node_k = nodes

        SG = self.system['graph']

        con_j = SG.degree(node_j) - 1
        con_k = SG.degree(node_k) - 1

        mult = con_j * con_k
        if mult == 0.0:
            return 'NA'

        # cases taken from the DREIDING paper (same cases, different force constants for UFF)
        # they are not done in order to save some lines, I don't know of a better way for doing
        # this besides a bunch of conditionals.
        if hyb_j == 'sp3' and hyb_k == 'sp3':
            # case (a)
            phi0 = 60.0
            n = 3.0
            V = 2.0
            # case (h)
            if els_j == 'O' and els_k == 'O':
                phi0 = 90.0
                n = 2.0
                V = 2.0
            elif els_j == 'S' and els_k == 'S':
                phi0 = 90.0
                n = 2.0
                V = 2.0

        elif (hyb_j in ('sp2', 'resonant') and hyb_k == 'sp3') or (hyb_k in ('sp2', 'resonant') and hyb_j == 'sp3'):
            # case (b)
            phi0 = 0.0
            n = 6.0
            V = 1.0

            if (hyb_j == 'sp3' and els_j == 'O') or (
                    hyb_k == 'sp3' and els_k == 'O'):
                # case (i)
                phi0 = 180.0
                n = 2.0
                V = 2.0
            # case (j) not needed for the current ToBaCCo MOFs

        elif hyb_j in ('sp2', 'resonant') and hyb_k in ('sp2', 'resonant'):
            if bond_order == 2.0:
                # case (c)
                phi0 = 180.0
                n = 2.0
                V = 45.0
            elif bond_order == 1.5:
                # case (d)
                phi0 = 180.0
                n = 2.0
                V = 25.0
            elif bond_order == 1.0 and (hyb_j != 'resonant' or hyb_k != 'resonant'):
                # case (e)
                phi0 = 180.0
                n = 2.0
                V = 5.0
            elif bond_order == 1.0 and (hyb_j == 'resonant' and hyb_k == 'resonant'):
                # case (f)
                phi0 = 180.0
                n = 2.0
                V = 10.0

        # case (g)
        elif hyb_j == 'sp1' or hyb_k == 'sp1':
            return 'NA'

        elif hyb_j == 'NA' or hyb_k == 'NA':
            return 'NA'

        # divide by multiplicity and halve to match UFF paper
        V /= mult
        V *= 0.5
        d = (n * phi0) + 180.0
        w = 0.0

        return ('charmm', V, int(n), int(d), w)

    def improper_parameters(self, fft_i):

        if fft_i in ('N_R', 'C_R', 'C_2'):

            K = 40.0 / 3.0
            omega0 = 0.0

        else:
            return None

        return ('fourier', K, omega0)

    def pair_parameters(self, charges=False):

        Dreiding_atom_parameters = self.args['FF_parameters']
        atom_types = self.atom_types
        params = {}
        comments = {}
        sb = 'dreiding'

        # determine style and special bonds
        if charges:
            style = 'lj/cut/coul/long'
            cutoff = 12.5
        else:
            style = 'lj/cut'
            cutoff = 12.5

        for a in atom_types:
            ID = atom_types[a]
            data = Dreiding_atom_parameters[a]
            sig_i = data[2] * (2**(-1.0 / 6.0))
            eps_i = data[3]
            params[ID] = (style, eps_i, sig_i)
            comments[ID] = [a, a]

        self.pair_data = {
            'params': params,
            'style': style,
            'special_bonds': sb,
            'comments': comments,
            'cutoff': cutoff}

    def enumerate_bonds(self):

        SG = self.system['graph']
        bond_order_dict = self.args['bond_orders']

        bonds = {}
        for e in SG.edges(data=True):

            i, j, data = e
            fft_i = SG.nodes[i]['force_field_type']
            fft_j = SG.nodes[j]['force_field_type']
            bond_type = data['bond_type']

            # look for the bond order, otherwise use the convention based on
            # the bond type
            try:
                bond_order = bond_order_dict[(fft_i, fft_j)]
            except KeyError:
                try:
                    bond_order = bond_order_dict[(fft_j, fft_i)]
                except KeyError:
                    bond_order = bond_order_dict[bond_type]

            bond = tuple(sorted([fft_i, fft_j]) + [bond_order])

            # add to list if bond type already exists, else add a new type
            try:
                bonds[bond].append((i, j))
            except KeyError:
                bonds[bond] = [(i, j)]

            data['bond_order'] = bond_order

        bond_params = {}
        bond_comments = {}
        all_bonds = {}
        ID = 0
        count = 0
        # index bonds by ID
        for b in bonds:

            ID += 1
            bond_order = float(b[2])
            bond = (b[0], b[1])
            params = self.bond_parameters(bond, bond_order)
            bond_params[ID] = list(params)
            bond_comments[ID] = list(bond) + ['bond order=' + str(bond_order)]
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
        # bonds = self.bond_data['all_bonds']
        # inv_bonds = dict((b, bt) for bt in bonds for b in bonds[bt])
        angles = {}

        for n in SG.nodes(data=True):

            name, data = n
            nbors = list(SG.neighbors(name))

            for comb in itertools.combinations(nbors, 2):

                j = name
                i, k = comb

                fft_i = SG.nodes[i]['force_field_type']
                fft_j = SG.nodes[j]['force_field_type']
                fft_k = SG.nodes[k]['force_field_type']

                sort_ik = sorted([(fft_i, i), (fft_k, k)], key=lambda x: x[0])
                fft_i, i = sort_ik[0]
                fft_k, k = sort_ik[1]

                # look up bond constants (don't need to calculate again, yay!)
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

            j, k = e[0:2]
            fft_j = SG.nodes[j]['force_field_type']
            fft_k = SG.nodes[k]['force_field_type']
            hyb_j = SG.nodes[j]['hybridization']
            hyb_k = SG.nodes[k]['hybridization']
            els_j = SG.nodes[j]['element_symbol']
            els_k = SG.nodes[k]['element_symbol']
            bond_order = e[2]['bond_order']
            nodes = (j, k)

            nbors_j = [n for n in SG.neighbors(j) if n != k]
            nbors_k = [n for n in SG.neighbors(k) if n != j]

            il_pairs = list(itertools.product(nbors_j, nbors_k))
            dihedral_list = [(p[0], j, k, p[1]) for p in il_pairs]

            bond = sorted([fft_j, fft_k])
            bond = (bond[0], bond[1], bond_order)
            hybridization = (hyb_j, hyb_k)
            element_symbols = (els_j, els_k)

            # here I calculate  parameters for each dihedral (I know) but I prefer identifying
            # those dihedrals before passing to the final dihedral data
            # construction.
            params = self.dihedral_parameters(
                bond, hybridization, element_symbols, nodes)

            if params != 'NA':
                try:
                    dihedrals[bond].extend(dihedral_list)
                except KeyError:
                    dihedrals[bond] = dihedral_list
                    dihedral_params[bond] = params

        all_dihedrals = {}
        dihedral_comments = {}
        indexed_dihedral_params = {}
        ID = 0
        count = 0
        for d in dihedrals:

            ID += 1
            dihedral = ('X', d[0], d[1], 'X')
            params = dihedral_params[d]
            all_dihedrals[ID] = dihedrals[d]
            indexed_dihedral_params[ID] = list(dihedral_params[d])
            dihedral_comments[ID] = list(
                dihedral) + ['bond order=' + str(d[2])]
            count += len(dihedrals[d])

        self.dihedral_data = {
            'all_dihedrals': all_dihedrals,
            'params': indexed_dihedral_params,
            'style': 'charmm',
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

                fft_i = data['force_field_type']
                j, k, LL = nbors

                # only need to consider one combination
                imps = [[i, j, k, LL]]

                try:
                    impropers[fft_i].extend(imps)
                except KeyError:
                    impropers[fft_i] = imps

        all_impropers = {}
        improper_params = {}
        improper_comments = {}
        ID = 0
        count = 0
        for i in impropers:

            fft_i = i

            params = self.improper_parameters(fft_i)

            if params is not None:
                ID += 1
                improper_params[ID] = list(params)
                improper_comments[ID] = [i[0], 'X', 'X', 'X']
                all_impropers[ID] = impropers[i]
                count += len(impropers[i])

        self.improper_data = {
            'all_impropers': all_impropers,
            'params': improper_params,
            'style': 'umbrella',
            'count': (
                count,
                len(all_impropers)),
            'comments': improper_comments}

    def compile_force_field(self, charges):

        self.type_atoms()
        self.pair_parameters(charges)
        self.enumerate_bonds()
        self.enumerate_angles()
        self.enumerate_dihedrals()
        self.enumerate_impropers()
