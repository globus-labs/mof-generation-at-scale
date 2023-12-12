from __future__ import print_function
# import numpy as np
import itertools
from . import atomic_data
from .force_field_construction import force_field

metals = atomic_data.metals
mass_key = atomic_data.mass_key


class MZHB(force_field):

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
            nbors = [a for a in SG.neighbors(name)]
            nbor_symbols = [SG.nodes[n]['element_symbol'] for n in nbors]
            mass = mass_key[element_symbol]

            if element_symbol == 'Si':
                ty = 'Si'
            elif element_symbol == 'O':
                ty = 'O'
            else:
                raise ValueError(
                    'No Nicholas type identified for ' +
                    element_symbol +
                    'with neighbors ' +
                    ' '.join(nbor_symbols))

            types.append((ty, element_symbol, mass))
            SG.nodes[name]['force_field_type'] = ty

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

    def pair_parameters(self, charges=True):

        SG = self.system['graph']
        atom_types = self.atom_types
        params = {}
        comments = {}

        style = 'lj/cut/coul/long'
        sb = 'lj/coul 1.0 1.0 1.0'

        for a in atom_types:

            ID = atom_types[a]

            if a == 'O':
                params[ID] = (style, 0.07472, 1.57689)
            elif a == 'Si':
                params[ID] = (style, 0.19924, 1.95998)

            comments[ID] = [a, a]

        for name, data in SG.nodes(data=True):

            if data['force_field_type'] == 'O':
                data['charge'] = -0.35
            elif data['force_field_type'] == 'Si':
                data['charge'] = 0.70
            else:
                pass

        self.pair_data = {
            'params': params,
            'style': style,
            'special_bonds': sb,
            'comments': comments}

    def bond_parameters(self, bond):

        i, j = bond

        # divide by two in LAMMPS
        if i == 'Si' and j == 'O':
            k_ij = 537.31 / 2.0
            r_ij = 1.620
        elif i == 'O' and j == 'Si':
            k_ij = 537.31 / 2.0
            r_ij = 1.620
        else:
            raise ValueError(
                'There is a non Si-O bond, which is not yet parametrized for Nicholas')

        return ('harmonic', k_ij, r_ij)

    def angle_parameters(self, angle):

        i, j, k = angle
        style = 'harmonic'

        if j == 'Si' and i == 'O' and k == 'O':

            # divide by two in LAMMPS
            K = 156.81169 / 2.0
            theta0 = 109.470

            return (style, K, theta0)

        elif j == 'O':

            # divide by two in LAMMPS
            K = 51.19440 / 2.0
            theta0 = 149.800

        return (style, K, theta0)

    def dihedral_parameters(self):

        # dihedrals are the same for everything
        pass

    def improper_parameters(self, fft_i, O_2_flag):

        # there are no impropers
        pass

    # Si-O bonds
    def enumerate_bonds(self):

        SG = self.system['graph']

        bonds = {}
        for e in SG.edges(data=True):

            i, j, data = e
            fft_i = SG.nodes[i]['force_field_type']
            fft_j = SG.nodes[j]['force_field_type']

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
            bond_comments[ID] = [b[0], b[1]]
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

                fft_i = SG.nodes[i]['force_field_type']
                fft_j = SG.nodes[j]['force_field_type']
                fft_k = SG.nodes[k]['force_field_type']

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
            angle_comments[ID] = [fft_i, fft_j, fft_k]
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

        pass

    def enumerate_impropers(self):

        pass

    def compile_force_field(self, charges=False):

        self.type_atoms()
        self.pair_parameters(charges)
        self.enumerate_bonds()
        self.enumerate_angles()
        self.enumerate_dihedrals()
        self.enumerate_impropers()
