from .main_conversion import serial_conversion
from .UFF4MOF_construction import UFF4MOF
# from .UFF_construction import UFF
# from .Dreiding_construction import Dreiding
# from .zeoliteFFs_construction import MZHB
# from .ZIFFF_construction import ZIFFF

if __name__ == '__main__':

    optional_arguments = {'force_field': UFF4MOF,
                          'ff_string': 'UFF4MOF',
                          'small_molecule_force_field': 'TIP4P',
                          'charges': False,
                          'replication': '1x1x1',
                          'read_cifs_pymatgen': False,
                          'add_molecule': None}

    # optional_arguments = {'force_field':UFF4MOF,
    # 					  'ff_string':'UFF4MOF',
    # 					  'small_molecule_force_field':'TIP4P',
    # 					  'charges':False,
    # 					  'replication':'1x1x1',
    # 					  'read_cifs_pymatgen':False,
    # 					  'add_molecule':('water','TIP3P_long',0)}

    # optional_arguments = {'force_field':UFF4MOF,
    # 				  'ff_string':'UFF4MOF',
    # 				  'small_molecule_force_field':None,
    # 				  'charges':False,
    # 				  'replication':'1x1x1',
    # 				  'read_cifs_pymatgen':False,
    # 				  'add_molecule':None}

    serial_conversion('unopt_cifs', **optional_arguments)
    # serial_conversion('check_cifs', **optional_arguments)
