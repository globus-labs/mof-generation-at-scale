from __future__ import print_function
from abc import abstractmethod

# import UFF4MOF_constants
from . import atomic_data

metals = atomic_data.metals
mass_key = atomic_data.mass_key


class force_field(object):

    """
        Abstract class for a force field, each force field will have an atom typing method,
        methods for finding all defined bonds, angles, dihedrals, and impropers, and methods
        for parametrizing all these defined interactions. The natural "base" object is a molecular
        graph of the system. The networkx graph is particularly convenient for this as all the
        needed information for each atom can be stored as node data.
    """

    def __init__(self, system, cutoff, args):

        self.system = system
        self.cutoff = cutoff
        self.args = args

    @abstractmethod
    def type_atoms(self):
        pass

    @abstractmethod
    def bond_parameters(self):
        pass

    @abstractmethod
    def angle_parameters(self):
        pass

    @abstractmethod
    def dihedral_parameters(self):
        pass

    @abstractmethod
    def improper_parameters(self):
        pass

    @abstractmethod
    def pair_parameters(self):
        pass

    @abstractmethod
    def enumerate_bonds(self):
        pass

    @abstractmethod
    def enumerate_angles(self):
        pass

    @abstractmethod
    def enumerate_dihedrals(self):
        pass

    @abstractmethod
    def enumerate_impropers(self):
        pass

    @abstractmethod
    def compile_force_field(self):
        pass
