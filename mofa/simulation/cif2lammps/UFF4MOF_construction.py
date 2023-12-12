from __future__ import print_function
import numpy as np
from numpy.linalg import norm
import math
import itertools
from itertools import permutations
from . import atomic_data
import warnings
from .force_field_construction import force_field
from .cif2system import PBC3DF_sym
from .superimposition import SVDSuperimposer

metals = atomic_data.metals
mass_key = atomic_data.mass_key

comparison_geometries = {

    'square_planar': np.array([[1.0, 0.0, 0.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, -1.0, 0.0]]),

    'tetrahedral': np.array([[1.0, 1.0, 0.0],
                             [1.0, 0.0, 1.0],
                             [0.0, 1.0, 1.0],
                             [0.0, 0.0, 0.0]])
}


def superimpose(a0, a1, count, max_permute=6):

    S = SVDSuperimposer()

    a0 = np.asarray(a0)
    a1 = np.asarray(a1)

    a0 -= np.average(a0, axis=0)
    a1 -= np.average(a1, axis=0)

    a0 = np.array([vec / norm(vec) if np.any(vec) else vec for vec in a0])
    a1 = np.array([vec / norm(vec) if np.any(vec) else vec for vec in a1])

    min_msd = (100.0, 'foo', 'bar')
    looper = list(permutations(a0))[0:max_permute]

    for LL in looper:

        p = np.asarray(LL)
        S.set(a1, p)
        S.run()
        msd = S.get_rms()

        if msd < min_msd[0]:
            rot, tran = S.get_rotran()
            min_msd = (msd, rot, tran)

    return min_msd[0]


def typing_loop(options, add, atom_type_dict):
    """
        types atoms in ambiguous cases, options should be ordered correctly
    """

    ty = None
    for option in options:

        try:
            ty = atom_type_dict[add + option]
            break

        except KeyError:
            continue

    if ty is not None:
        return add + option
    else:
        return None


class UFF4MOF(force_field):

    def __init__(self, system, cutoff, args):

        self.system = system
        self.cutoff = cutoff
        self.args = args

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

    def type_atoms(self):

        UFF4MOF_atom_parameters = self.args['FF_parameters']
        SG = self.system['graph']
        types = []
        count = 0

        for atom in SG.nodes(data=True):

            ty = None
            name, inf = atom
            element_symbol = inf['element_symbol']
            nbors = [a for a in SG.neighbors(name)]
            nbor_symbols = [SG.nodes[n]['element_symbol'] for n in nbors]
            bond_types = [
                SG.get_edge_data(
                    name,
                    n)['bond_type'] for n in nbors]
            mass = mass_key[element_symbol]

            if len(nbors) > 1:

                # 4 connected metals require shape matching
                if (element_symbol in metals and len(nbors) == 4):

                    count += 1

                    comp_coords = []
                    for n in nbors:

                        dist_n, sym_n = PBC3DF_sym(
                            SG.nodes[n]['fractional_position'], inf['fractional_position'])
                        dist_n = np.dot(self.unit_cell, dist_n)
                        fcoord = SG.nodes[n]['fractional_position'] + sym_n
                        comp_coords.append(np.dot(self.unit_cell, fcoord))

                    comp_coords = np.array(comp_coords)
                    dist_square = superimpose(
                        comp_coords, comparison_geometries['square_planar'], count)
                    dist_tetrahedral = superimpose(
                        comp_coords, comparison_geometries['tetrahedral'], count)

                else:

                    angles = []
                    for n0, n1 in itertools.combinations(nbors, 2):

                        dist_j, sym_j = PBC3DF_sym(
                            SG.nodes[n0]['fractional_position'], inf['fractional_position'])
                        dist_k, sym_k = PBC3DF_sym(
                            SG.nodes[n1]['fractional_position'], inf['fractional_position'])

                        dist_j = np.dot(self.unit_cell, dist_j)

                        cosine_angle = np.dot(
                            dist_j, dist_k) / (np.linalg.norm(dist_j) * np.linalg.norm(dist_k))

                        if cosine_angle > 1:
                            cosine_angle = 1
                        elif cosine_angle < -1:
                            cosine_angle = -1

                        ang = (180.0 / np.pi) * np.arccos(cosine_angle)

                        angles.append(ang)

                    angles = np.array(angles)
                    dist_linear = min(np.abs(angles - 180.0))
                    dist_triangle = min(np.abs(angles - 120.0))
                    dist_square = min(min(np.abs(angles - 90.0)),
                                      min(np.abs(angles - 180.0)))
                    dist_corner = min(np.abs(angles - 90.0))
                    dist_tetrahedral = min(np.abs(angles - 109.47))

            if 'A' in bond_types and element_symbol not in (
                    'O', 'H') and element_symbol not in metals:
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
                        hyb = 'sp' + str(len(nbors) - 1)
                    else:
                        ty = element_symbol + str(len(nbors) - 1)
                        hyb = 'sp' + str(len(nbors) - 1)
                # Group 7
                elif element_symbol in ('N', 'P'):
                    if element_symbol == 'N':
                        if len(nbors) < 4:
                            # special case for -NO2
                            if sorted(nbor_symbols) == ['C', 'O', 'O']:
                                ty = element_symbol + '_R'
                                hyb = 'sp2'
                            else:
                                ty = element_symbol + '_' + str(len(nbors))
                                hyb = 'sp' + str(len(nbors))
                        else:
                            ty = 'N_3'
                    elif element_symbol == 'P':
                        if dist_square < dist_tetrahedral:
                            ty = 'P_3+3'
                        else:
                            ty = 'P_3+5'
                # Group 8
                elif element_symbol in ('O', 'S'):
                    if element_symbol == 'O':
                        # =O for example
                        if len(nbors) == 1:
                            ty = 'O_1'
                            hyb = 'sp1'
                        # -OH, for example
                        elif len(nbors) == 2 and 'A' not in bond_types and 'D' not in bond_types and not any(i in metals for i in nbor_symbols):
                            ty = 'O_3'
                            hyb = 'sp3'
                        # coordinated solvent, same parameters as O_3, but
                        # different name to modulate bond orders
                        elif len(nbors) in (2, 3) and len([i for i in nbor_symbols if i in metals]) == 1 and 'H' in nbor_symbols:
                            ty = 'O_3_M'
                            hyb = 'sp3'
                        # furan oxygen, for example
                        elif len(nbors) == 2 and 'A' in bond_types and not any(i in metals for i in nbor_symbols):
                            ty = 'O_R'
                            hyb = 'sp2'
                        # carboxyllic oxygen
                        elif len(nbors) == 2 and 'D' in bond_types and not any(i in metals for i in nbor_symbols):
                            ty = 'O_2'
                            hyb = 'sp2'
                        # carboxylate oxygen bound to metal node, same
                        # parameters as O_2, but different name to modulate
                        # bond orders
                        elif len(nbors) == 2 and any(i in metals for i in nbor_symbols) and 'C' in nbor_symbols:
                            ty = 'O_2_M'
                            hyb = 'sp2'
                        # 2 connected oxygens bonded to metals
                        elif len(nbors) == 2 and len([i for i in nbor_symbols if i in metals]) == 1:
                            ty = 'O_3'
                            hyb = 'sp2'
                        elif len(nbors) == 2 and len([i for i in nbor_symbols if i in metals]) == 2:
                            ty = 'O_2_z'
                            hyb = 'sp2'
                        # 3 connected oxygens bonded to metals
                        elif len(nbors) == 3 and len([i for i in nbor_symbols if i in metals]) > 1:
                            # trigonal geometry
                            if dist_triangle < dist_tetrahedral and not any(
                                    i in ['Zr', 'Eu', 'Tb', 'U'] for i in nbor_symbols):
                                ty = 'O_2_z'
                                hyb = 'sp2'
                            # sometimes oxygens in Zr6 and analagous nodes can
                            # have near trigonal geometry, still want O_3_f,
                            # however
                            elif dist_triangle < dist_tetrahedral and any(i in ['Zr', 'Eu', 'Tb', 'U'] for i in nbor_symbols):
                                ty = 'O_3_f'
                                hyb = 'sp2'
                            # tetrahedral-like geometry
                            elif dist_tetrahedral < dist_triangle and any(i in ['Zr', 'Eu', 'Tb', 'U'] for i in nbor_symbols):
                                ty = 'O_3_f'
                                hyb = 'sp3'
                        # 4 connected oxygens bonded to metals
                        elif len(nbors) == 4 and any(i in metals for i in nbor_symbols):
                            ty = 'O_3_f'
                            hyb = 'sp3'

                    # sulfur case is simple
                    elif element_symbol == 'S':
                        # -SH like patterns
                        if len(nbors) == 2:
                            ty = 'S_' + str(len(nbors) + 1)
                            hyb = 'sp' + str(len(nbors) + 1)
                        # trigonal S
                        elif len(nbors) == 3:
                            ty = 'S_2'
                            hyb = 'sp2'
                        # tetrahedral S connected to metals
                        elif len(nbors) > 3 and any(i in metals for i in nbor_symbols):
                            ty = 'S_3_f'
                            hyb = 'sp3'
                        # tetrahedral S connected to non-metals
                        elif len(nbors) > 3 and not any(i in metals for i in nbor_symbols):
                            ty = 'S_3+6'
                            hyb = 'sp3'
                # Group 9
                elif element_symbol in ('F', 'Cl', 'Br') and len(nbor_symbols) in (0, 1, 2, 4):
                    if element_symbol != 'Cl':
                        if len(element_symbol) == 1:
                            ty = element_symbol + '_'
                        else:
                            ty = element_symbol
                        hyb = 'sp1'
                    # some Cl have 90 degree angles in CoRE MOFs
                    else:
                        if len(nbor_symbols) > 0:
                            if dist_corner <= dist_linear:
                                ty = 'Cl_f'
                            elif dist_linear < dist_corner:
                                ty = 'Cl'
                        else:
                            ty = 'Cl'

                # metals
                elif element_symbol in metals and element_symbol not in ('As', 'Bi', 'Tl', 'Sb', 'At', 'Cs', 'Fr', 'Ni', 'Rb'):

                    hyb = 'NA'

                    # symbol to add to type
                    if len(element_symbol) == 1:
                        add_symbol = element_symbol + '_'
                    else:
                        add_symbol = element_symbol

                    # 2 connected, linear
                    if len(nbors) == 2 and dist_linear < 60.0:
                        options = (
                            '1f1', '4f2', '4+2', '6f3', '6+3', '6+2', '6+4')
                        ty = typing_loop(
                            options, add_symbol, UFF4MOF_atom_parameters)

                    # incomplete square planar
                    elif len(nbors) == 3 and dist_square < min(dist_tetrahedral, dist_triangle):
                        options = ('4f2', '4+2')
                        ty = typing_loop(
                            options, add_symbol, UFF4MOF_atom_parameters)

                    # incomplete tetrahedron
                    elif len(nbors) == 3 and dist_tetrahedral < min(dist_square, dist_triangle):
                        options = ('3f2', '3+2')
                        ty = typing_loop(
                            options, add_symbol, UFF4MOF_atom_parameters)

                    # trigonal, only Cu, Zn, Ag
                    elif len(nbors) == 3 and dist_triangle < min(dist_square, dist_tetrahedral):
                        options = ['2f2']
                        ty = typing_loop(
                            options, add_symbol, UFF4MOF_atom_parameters)

                    # 4 connected, square planar
                    elif len(nbors) == 4 and dist_square < dist_tetrahedral:
                        options = ('4f2', '4+2', '6f3', '6+3', '6+2', '6+4')
                        ty = typing_loop(
                            options, add_symbol, UFF4MOF_atom_parameters)

                    # 4 connected, tetrahedral 3+3 does not apply for As, Sb,
                    # Tl, Bi
                    elif len(nbors) == 4 and (dist_tetrahedral < dist_square):
                        options = ('3f2', '3f4', '3+1', '3+2',
                                   '3+3', '3+4', '3+5', '3+6')
                        ty = typing_loop(
                            options, add_symbol, UFF4MOF_atom_parameters)

                    # paddlewheels, 5 neighbors if bare, 6 neighbors if
                    # pillared, one should be another metal
                    elif len(nbors) in (5, 6) and any(i in metals for i in nbor_symbols):
                        if (dist_square < dist_tetrahedral):
                            options = (
                                '4f2', '4+2', '6f3', '6+3', '6+2', '6+4')
                            ty = typing_loop(
                                options, add_symbol, UFF4MOF_atom_parameters)
                        else:
                            options = (
                                '4f2', '4+2', '6f3', '6+3', '6+2', '6+4')
                            ty = typing_loop(
                                options, add_symbol, UFF4MOF_atom_parameters)
                            message = 'There is a ' + element_symbol + \
                                ' that has a near tetrahedral angle typed as ' + ty + '\n'
                            message += 'The neighbors are ' + \
                                ' '.join(nbor_symbols)
                            warnings.warn(message)

                    # M3O(CO2H)6 metals, e.g. MIL-100, paddlewheel options are
                    # secondary, followed by 8f4 (should give nearly correct
                    # geometry)
                    elif len(nbors) in (5, 6) and not any(i in metals for i in nbor_symbols) and (dist_square < dist_tetrahedral):
                        options = ('6f3', '6+2', '6+3', '6+4', '4f2', '4+2')
                        ty = typing_loop(
                            options, add_symbol, UFF4MOF_atom_parameters)

                    # 5,6 connected, with near-tetrahedral angles
                    elif len(nbors) in (5, 6) and not any(i in metals for i in nbor_symbols) and (dist_tetrahedral < dist_square):
                        options = (
                            '8f4',
                            '3f2',
                            '3f4',
                            '3+1',
                            '3+2',
                            '3+3',
                            '3+4',
                            '3+5',
                            '3+6')
                        ty = typing_loop(
                            options, add_symbol, UFF4MOF_atom_parameters)

                    # highly connected metals (max of 12 neighbors)
                    elif 7 <= len(nbors) <= 14:
                        options = ['8f4']
                        ty = typing_loop(
                            options, add_symbol, UFF4MOF_atom_parameters)

                # only one type for Bi
                elif element_symbol in ('As', 'Bi', 'Tl', 'Sb'):
                    ty = element_symbol + '3+3'
                    hyb = 'NA'
                elif element_symbol == 'At':
                    ty = 'At'
                    hyb = 'NA'
                elif element_symbol == 'Cs':
                    ty = 'Cs'
                    hyb = 'NA'
                elif element_symbol == 'Fr':
                    ty = 'Fr'
                    hyb = 'NA'
                elif element_symbol == 'Ni':
                    ty = 'Ni4+2'
                    hyb = 'NA'
                elif element_symbol == 'Rb':
                    ty = 'Rb'
                    hyb = 'NA'

            types.append((ty, element_symbol, mass))
            SG.nodes[name]['force_field_type'] = ty
            SG.nodes[name]['hybridization'] = hyb

            if ty is None:
                raise ValueError(
                    'No UFF4MOF type identified for atom ' +
                    element_symbol +
                    ' with neighbors ' +
                    ' '.join(nbor_symbols))
            elif ty == 'C_R' and len(nbor_symbols) > 3:
                raise ValueError(
                    'Too many neighbors for aromatic carbon ' +
                    element_symbol +
                    ' with neighbors ' +
                    ' '.join(nbor_symbols))

        types = set(types)
        types = sorted(types, key=lambda x: x[0])
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

        UFF4MOF_atom_parameters = self.args['FF_parameters']

        i, j = bond
        params_i = UFF4MOF_atom_parameters[i]
        params_j = UFF4MOF_atom_parameters[j]

        if i == 'Zr8f4' and j == 'Zr8f4':
            return ('zero', 'nocoeff')

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

        UFF4MOF_atom_parameters = self.args['FF_parameters']

        i, j, k = angle
        angle_style = 'cosine/periodic'

        params_i = UFF4MOF_atom_parameters[i]
        params_j = UFF4MOF_atom_parameters[j]
        params_k = UFF4MOF_atom_parameters[k]

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
            n = 'NA'
            b = 'NA'

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

        # the 1/2 scaling is needed to correct the LAMMPS angle energy calculation
        # angle energy for angle_style cosine/periodic is multiplied by 2 in LAMMPS for some reason
        # see https://github.com/lammps/lammps/blob/master/src/MOLECULE/angle_cosine_periodic.cpp, line 140
        # the 1/2 factor for the UFF fourier angles should be included here as
        # it is not included in LAMMPS
        K *= 0.5

        return (angle_style, K, b, n)

    def dihedral_parameters(self, bond, hybridization, element_symbols, nodes):

        fft_j, fft_k, bond_order = bond
        hyb_j, hyb_k = hybridization
        els_j, els_k = element_symbols
        node_j, node_k = nodes

        SG = self.system['graph']
        UFF4MOF_atom_parameters = self.args['FF_parameters']

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
            V_j = UFF4MOF_atom_parameters[fft_j][6]
            V_k = UFF4MOF_atom_parameters[fft_k][6]
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
                U_j = UFF4MOF_atom_parameters[fft_j][6]
                U_k = UFF4MOF_atom_parameters[fft_k][6]
                V = 5 * np.sqrt(U_j * U_k) * (1.0 + 4.18 * np.log(bond_order))
            elif hyb_k == 'sp3' and els_k in ('O', 'S'):
                phi0 = 180.0
                n = 2.0
                U_j = UFF4MOF_atom_parameters[fft_j][6]
                U_k = UFF4MOF_atom_parameters[fft_k][6]
                V = 5 * np.sqrt(U_j * U_k) * (1.0 + 4.18 * np.log(bond_order))
            # case (j) not needed for the current ToBaCCo MOFs

        # case (c, d, e, f)
        elif hyb_j in ('sp2', 'resonant') and hyb_k in ('sp2', 'resonant'):
            phi0 = 180.0
            n = 2.0
            U_j = UFF4MOF_atom_parameters[fft_j][6]
            U_k = UFF4MOF_atom_parameters[fft_k][6]
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

        UFF4MOF_atom_parameters = self.args['FF_parameters']
        atom_types = self.atom_types
        params = {}
        comments = {}

        # determine style and special bonds
        if charges:
            style = 'lj/cut/coul/long'
            sb = 'lj 0.0 0.0 1.0 coul 0.0 0.0 0.0'
        else:
            style = 'lj/cut'
            sb = 'lj 0.0 0.0 1.0'

        for a in atom_types:
            ID = atom_types[a]
            data = UFF4MOF_atom_parameters[a]
            x_i = data[2] * (2**(-1.0 / 6.0))
            D_i = data[3]
            params[ID] = (style, D_i, x_i)
            comments[ID] = [a, a]

        self.pair_data = {
            'params': params,
            'style': style,
            'special_bonds': sb,
            'comments': comments}

    def enumerate_bonds(self):

        SG = self.system['graph']
        bond_order_dict = self.args['bond_orders']

        bonds = {}
        for e in SG.edges(data=True):

            i, j, data = e
            fft_i = SG.nodes[i]['force_field_type']
            fft_j = SG.nodes[j]['force_field_type']
            bond_type = data['bond_type']
            esi = SG.nodes[i]['element_symbol']
            esj = SG.nodes[j]['element_symbol']

            # look for the bond order, otherwise use the convention based on
            # the bond type
            try:
                bond_order = bond_order_dict[(fft_i, fft_j)]
            except KeyError:
                try:
                    bond_order = bond_order_dict[(fft_j, fft_i)]
                except KeyError:
                    # half for metal-nonmetal
                    if any(
                        a in metals for a in (
                            esi,
                            esj)) and not all(
                        a in metals for a in (
                            esi,
                            esj)):
                        bond_order = 0.5
                    # quarter for metal-metal
                    elif all(a in metals for a in (esi, esj)):
                        bond_order = 0.25
                    # use bond order
                    else:
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
        styles = []
        # index bonds by ID
        for b in bonds:

            ID += 1
            bond_order = b[2]
            bond = (b[0], b[1])
            params = self.bond_parameters(bond, float(bond_order))
            styles.append(params[0])
            bond_params[ID] = list(params)
            bond_comments[ID] = list(bond) + ['bond order=' + str(bond_order)]
            all_bonds[ID] = bonds[b]
            count += len(bonds[b])

        styles = set(styles)
        if len(styles) == 1:
            style = list(styles)[0]
        else:
            style = 'hybrid ' + ' '.join(styles)

        self.bond_data = {
            'all_bonds': all_bonds,
            'params': bond_params,
            'style': style,
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

                try:
                    r_ij = bond_params[bond_type_ij][2]
                    r_jk = bond_params[bond_type_jk][2]
                except IndexError:
                    continue

                angle = sorted((fft_i, fft_k))
                sorted_rs = sorted((r_ij, r_jk))

                # angle = (angle[0], fft_j, angle[1], r_ij, r_jk)
                angle = (angle[0], fft_j, angle[1], sorted_rs[0], sorted_rs[1])

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
