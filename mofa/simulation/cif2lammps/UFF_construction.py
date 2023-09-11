from __future__ import print_function
import numpy as np
import math
import itertools
from . import atomic_data
from .force_field_construction import force_field

metals = atomic_data.metals
mass_key = atomic_data.mass_key


class UFF(force_field):

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

            # Atom typing for UFF, this can be made much more robust with pattern matching,
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
                    if len(element_symbol) == 1:
                        ty = element_symbol + '_' + str(len(nbors) - 1)
                    else:
                        ty = element_symbol + str(len(nbors) - 1)
                    hyb = 'sp' + str(len(nbors) - 1)
                # Group 7
                elif element_symbol in ('N'):
                    ty = element_symbol + '_' + str(len(nbors))
                    hyb = 'sp' + str(len(nbors))
                # Group 8
                elif element_symbol in ('O', 'S'):
                    # oxygen case is complex with the UFF4MOF oxygen types
                    if element_symbol == 'O':
                        # =O for example
                        if len(nbors) == 1:
                            ty = 'O_1'
                            hyb = 'sp1'
                        # -OH, for example
                        elif len(nbors) == 2 and 'A' not in bond_types and 'D' not in bond_types and not any(i in metals for i in nbor_symbols):
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
                        elif len(nbors) == 3 and all(i in metals for i in nbor_symbols) and 'Zr' not in nbor_symbols:
                            ty = 'O_2_M'
                            hyb = 'sp2'
                        elif len(nbors) == 3 and all(i in metals for i in nbor_symbols) and 'Zr' in nbor_symbols:
                            ty = 'O_3_M'
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
                    if len(element_symbol) == 1:
                        ty = element_symbol + '_'
                    else:
                        ty = element_symbol
                    hyb = 'sp1'
                # Metals
                elif element_symbol in metals:
                    # Cu paddlewheel, just changed equilibrium angle of Cu3+1
                    # to 90.0
                    if len(nbors) == 5 and element_symbol == 'Cu' and any(
                            i in metals for i in nbor_symbols):
                        ty = element_symbol + '4+1'
                        hyb = 'NA'
                    # M3O(CO2H)6 metals, e.g. MIL-100
                    elif len(nbors) in (5, 6) and element_symbol in ('Al', 'Sc', 'V', 'Mn', 'Fe', 'Cr') and not any(i in metals for i in nbor_symbols):
                        ty = element_symbol + '6+3'
                        if element_symbol == 'V':
                            ty = 'V_6+3'
                        hyb = 'NA'
                    # IRMOF-1 node
                    elif len(nbors) == 4 and element_symbol == 'Zn':
                        ty = 'Zn3+2'
                        hyb = 'NA'
                    # Zr node
                    elif len(nbors) in (7, 8) and element_symbol == 'Zr':
                        ty = 'Zr3+4'
                        hyb = 'NA'
                # if no type can be identified
                else:
                    raise ValueError(
                        'No UFF type identified for ' +
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

        # SG = self.system['graph']
        UFF_atom_parameters = self.args['FF_parameters']

        i, j = bond
        params_i = UFF_atom_parameters[i]
        params_j = UFF_atom_parameters[j]

        r0_i, theta0_i, x1_i, D1_i, zeta_i, Z1_i, V_i, X_i = params_i
        r0_j, theta0_j, x1_j, D1_j, zeta_j, Z1_j, V_j, X_j = params_j

        # bond-order correction
        rbo = -0.1332 * (r0_i + r0_j) * np.log(bond_order)
        # electronegativity correction
        ren = r0_i * r0_j * (((np.sqrt(X_i) - np.sqrt(X_j))**2)
                             ) / (X_i * r0_i + X_j * r0_j)
        # equilibrium distance
        r_ij = r0_i + r0_j + rbo - ren
        r_ij3 = r_ij * r_ij * r_ij
        # force constant (1/2 factor should be included here for LAMMPS)
        k_ij = 0.5 * 664.12 * ((Z1_i * Z1_j) / r_ij3)

        return ('harmonic', k_ij, r_ij)

    def angle_parameters(self, angle, r_ij, r_jk):

        UFF_atom_parameters = self.args['FF_parameters']

        i, j, k = angle
        angle_style = 'cosine/periodic'

        params_i = UFF_atom_parameters[i]
        params_j = UFF_atom_parameters[j]
        params_k = UFF_atom_parameters[k]

        r0_i, theta0_i, x1_i, D1_i, zeta_i, Z1_i, V_i, X_i = params_i
        r0_j, theta0_j, x1_j, D1_j, zeta_j, Z1_j, V_j, X_j = params_j
        r0_k, theta0_k, x1_k, D1_k, zeta_k, Z1_k, V_k, X_k = params_k

        # linear
        if theta0_j == 180.0:
            n = 1
            b = 1
        # trigonal planar
        elif theta0_j == 120.0:
            n = 3
            b = -1
        # square planar or octahedral
        elif theta0_j == 90.0:
            n = 4
            b = 1
        # general non-linear
        else:
            b = 'NA'
            n = 'NA'

        cosT0 = np.cos(math.radians(theta0_j))
        sinT0 = np.sin(math.radians(theta0_j))

        r_ik = np.sqrt(r_ij**2.0 + r_jk**2.0 - 2.0 * r_ij * r_jk * cosT0)
        # force constant
        K = ((664.12 * Z1_i * Z1_k) / (r_ik**5.0)) * \
            (3.0 * r_ij * r_jk * (1.0 - cosT0**2.0) - r_ik**2.0 * cosT0)

        # general non-linear
        if theta0_j not in (90.0, 120.0, 180.0):

            angle_style = 'fourier'
            C2 = 1.0 / (4 * sinT0**2)
            C1 = -4 * C2 * cosT0
            C0 = C2 * (2 * cosT0**2 + 1)

            return (angle_style, K, C0, C1, C2)

        # this is needed to correct the LAMMPS angle energy calculation
        K *= 0.5

        return (angle_style, K, b, n)

    def dihedral_parameters(self, bond, hybridization, element_symbols, nodes):

        fft_j, fft_k, bond_order = bond
        hyb_j, hyb_k = hybridization
        els_j, els_k = element_symbols
        node_j, node_k = nodes

        SG = self.system['graph']
        UFF_atom_parameters = self.args['FF_parameters']

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
            V_j = UFF_atom_parameters[fft_j][6]
            V_k = UFF_atom_parameters[fft_k][6]
            V = np.sqrt(V_j * V_k)
            # case (h)
            if els_j == 'O' and els_k == 'O':
                phi0 = 90.0
                n = 2.0
                V = 2.0
            elif els_j == 'S' and els_k == 'S':
                phi0 = 90.0
                n = 2.0
                V = 6.8

        elif (hyb_j in ('sp2', 'resonant') and hyb_k == 'sp3') or (hyb_k in ('sp2', 'resonant') and hyb_j == 'sp3'):
            # case (b)
            phi0 = 180.0
            n = 6.0
            V = 2.0
            # case (i)
            if hyb_j == 'sp3' and els_j in ('O', 'S'):
                phi0 = 180.0
                n = 2.0
                U_j = UFF_atom_parameters[fft_j][6]
                U_k = UFF_atom_parameters[fft_k][6]
                V = 5 * np.sqrt(U_j * U_k) * (1.0 + 4.18 * np.log(bond_order))
            elif hyb_k == 'sp3' and els_k in ('O', 'S'):
                phi0 = 180.0
                n = 2.0
                U_j = UFF_atom_parameters[fft_j][6]
                U_k = UFF_atom_parameters[fft_k][6]
                V = 5 * np.sqrt(U_j * U_k) * (1.0 + 4.18 * np.log(bond_order))
            # case (j) not needed for the current ToBaCCo MOFs

        # case (c, d, e, f)
        elif hyb_j in ('sp2', 'resonant') and hyb_k in ('sp2', 'resonant'):
            phi0 = 180.0
            n = 2.0
            U_j = UFF_atom_parameters[fft_j][6]
            U_k = UFF_atom_parameters[fft_k][6]
            V = 5 * np.sqrt(U_j * U_k) * (1.0 + 4.18 * np.log(bond_order))

        # case (g)
        elif hyb_j == 'sp1' or hyb_k == 'sp1':
            return 'NA'

        elif hyb_j == 'NA' or hyb_k == 'NA':
            return 'NA'

        # divide by multiplicity and halve to match UFF paper
        V /= mult
        V *= 0.5
        d = -1.0 * np.cos(math.radians(n * phi0))

        return ('harmonic', V, int(d), int(n))

    def improper_parameters(self, fft_i, O_2_flag):

        if fft_i in ('N_R', 'C_R', 'C_2'):

            # constants for C_R and N_R
            C0 = 1.0
            C1 = -1.0
            C2 = 0.0
            K = 6.0 / 3.0
            al = 1

            # constants for bound O_2
            if O_2_flag:
                K = 50.0 / 3.0

        else:
            return None

        return ('fourier', K, C0, C1, C2, al)

    def pair_parameters(self, charges=False):

        UFF_atom_parameters = self.args['FF_parameters']
        atom_types = self.atom_types
        params = {}
        comments = {}

        # determine style and special bonds
        if charges:
            style = 'lj/cut/coul/long'
            cutoff = 12.5
            sb = 'lj/coul 0.0 0.0 1.0'
        else:
            style = 'lj/cut'
            cutoff = 12.5
            sb = 'lj 0.0 0.0 1.0'

        for a in atom_types:
            ID = atom_types[a]
            data = UFF_atom_parameters[a]
            x_i = data[2] * (2**(-1.0 / 6.0))
            D_i = data[3]
            params[ID] = (style, D_i, x_i)
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
        bonds = self.bond_data['all_bonds']
        bond_params = self.bond_data['params']
        inv_bonds = dict((b, bt) for bt in bonds for b in bonds[bt])
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

                octa_metals = (
                    'Al6+3',
                    'Sc6+3',
                    'Ti4+2',
                    'V_4+2',
                    'V_6+3',
                    'Cr4+2',
                    'Cr6f3',
                    'Mn6+3',
                    'Mn4+2',
                    'Fe6+3',
                    'Fe4+2',
                    'Co4+2',
                    'Cu4+2',
                    'Zn4+2')

                if fft_j in octa_metals:
                    i_coord = SG.nodes[i]['cartesian_position']
                    j_coord = SG.nodes[j]['cartesian_position']
                    k_coord = SG.nodes[k]['cartesian_position']
                    ij = i_coord - j_coord
                    jk = j_coord - k_coord
                    cosine_angle = np.dot(
                        ij, jk) / (np.linalg.norm(ij) * np.linalg.norm(jk))
                    angle = (180.0 / np.pi) * np.arccos(cosine_angle)

                sort_ik = sorted([(fft_i, i), (fft_k, k)], key=lambda x: x[0])
                fft_i, i = sort_ik[0]
                fft_k, k = sort_ik[1]

                # look up bond constants (don't need to calculate again, yay!)
                try:
                    bond_type_ij = inv_bonds[(i, j)]
                except KeyError:
                    bond_type_ij = inv_bonds[(j, i)]
                try:
                    bond_type_jk = inv_bonds[(j, k)]
                except KeyError:
                    bond_type_jk = inv_bonds[(k, j)]

                r_ij = bond_params[bond_type_ij][2]
                r_jk = bond_params[bond_type_jk][2]

                angle = sorted((fft_i, fft_k))
                angle = (angle[0], fft_j, angle[1], r_ij, r_jk)

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
            fft_i, fft_j, fft_k, r_ij, r_jk = a
            angle = (fft_i, fft_j, fft_k)
            params = self.angle_parameters(angle, r_ij, r_jk)
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

                fft_i = data['force_field_type']
                fft_nbors = tuple(
                    sorted([SG.nodes[m]['force_field_type'] for m in nbors]))
                O_2_flag = False
                # force constant is much larger if j,k, or l is O_2
                if 'O_2' in fft_nbors or 'O_2_M' in fft_nbors:
                    O_2_flag = True
                j, k, LL = nbors

                # only need to consider one combination
                imps = [[i, j, k, LL]]

                try:
                    impropers[(fft_i, O_2_flag)].extend(imps)
                except KeyError:
                    impropers[(fft_i, O_2_flag)] = imps

        all_impropers = {}
        improper_params = {}
        improper_comments = {}
        ID = 0
        count = 0
        for i in impropers:

            fft_i, O_2_flag = i

            params = self.improper_parameters(fft_i, O_2_flag)

            if params is not None:
                ID += 1
                improper_params[ID] = list(params)
                improper_comments[ID] = [
                    i[0], 'X', 'X', 'X', 'O_2 present=' + str(O_2_flag)]
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
