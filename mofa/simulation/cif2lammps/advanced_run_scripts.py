from .main_conversion import single_conversion, serial_conversion
from .UFF4MOF_construction import UFF4MOF
# from .UFF_construction import UFF
# from .Dreiding_construction import Dreiding
# from .zeoliteFFs_construction import MZHB
# from .ZIFFF_construction import ZIFFF
import glob
import os


def test_run():

    optional_arguments = {'force_field': UFF4MOF,
                          'ff_string': 'UFF4MOF',
                          'small_molecule_force_field': None,
                          'charges': True,
                          'replication': '',
                          'read_cifs_pymatgen': False,
                          'add_molecule': None,
                          'small_molecule_file': None}

    # optional_arguments = {'force_field':UFF4MOF,
    # 'ff_string':'UFF4MOF',
    # 'small_molecule_force_field':'TIP4P',
    # 'charges':False,
    # 'replication':'1x1x1',
    # 'read_cifs_pymatgen':False,
    # 'add_molecule':('water','TIP3P_long',0)}

    # optional_arguments = {'force_field':UFF4MOF,
    # 'ff_string':'UFF4MOF',
    # 'small_molecule_force_field':'Ions',
    # 'charges':False,
    # 'replication':'',
    # 'read_cifs_pymatgen':False,
    # 'add_molecule':None}

    serial_conversion('unopt_cifs', **optional_arguments)
    # serial_conversion('check_cifs', **optional_arguments)


def multiple_sm_loadings(direc):

    cifs = glob.glob(direc + os.sep + '*.cif')
    pdbs = glob.glob(direc + os.sep + '*.pdb')

    pairs = []

    for cif in cifs:
        for pdb in pdbs:

            cifname = cif.split('/')[-1].split('.')[0]
            pdbname = pdb.split('/')[-1].split('.')[0]

            if cifname == pdbname:

                pairs.append((cif, pdb))

    for pair in pairs:

        OA = {'force_field': UFF4MOF,
              'ff_string': 'UFF4MOF',
              'small_molecule_force_field': 'TIP4P_2005_long',
              'charges': True,
              'replication': '',
              'read_cifs_pymatgen': False,
              'add_molecule': None,
              'small_molecule_file': pair[1].split('/')[-1]}

        single_conversion(pair[0], **OA)


if __name__ == '__main__':

    test_run()
    # multiple_sm_loadings('unopt_cifs')
