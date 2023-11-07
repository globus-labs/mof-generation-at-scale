from __future__ import print_function
# , duplicate_system, write_cif_from_system
from .cif2system import initialize_system, replication_determination
from . import atomic_data
import os
import numpy as np
import math

# from .UFF4MOF_construction import UFF4MOF
from . import UFF4MOF_constants

# from .UFF_construction import UFF
from . import UFF_constants

# from .Dreiding_construction import Dreiding
from . import Dreiding_constants


UFF4MOF_atom_parameters = UFF4MOF_constants.UFF4MOF_atom_parameters
UFF4MOF_bond_orders_0 = UFF4MOF_constants.UFF4MOF_bond_orders_0

UFF_atom_parameters = UFF_constants.UFF_atom_parameters
UFF_bond_orders_0 = UFF_constants.UFF_bond_orders_0

Dreiding_atom_parameters = Dreiding_constants.Dreiding_atom_parameters
Dreiding_bond_orders_0 = Dreiding_constants.Dreiding_bond_orders_0

mass_key = atomic_data.mass_key

# add more force field classes here as they are made

# this is a placeholder script for conversion to GULP, currently it is just used
# for validating LAMMPS UFF4MOF calculations, it will be exanpded for more
# general GULP usage


def isfloat(value):
    """
        determines if a value is a float
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


mass_key = atomic_data.mass_key


def GULP_inputs(args):

    gulp_bond_types = {
        0.25: 'quarter',
        0.5: 'half',
        1.0: '',
        1.5: 'resonant',
        2.0: '',
        3.0: ''}

    cifname, force_field, outdir, charges, replication, noautobond = args
    FF_args = {
        'FF_parameters': UFF4MOF_atom_parameters,
        'bond_orders': UFF4MOF_bond_orders_0}
    cutoff = 12.5

    system = initialize_system(cifname, charges=charges)
    system, replication = replication_determination(
        system, replication, cutoff)
    FF = force_field(system, cutoff, FF_args)
    FF.compile_force_field(charges=charges)

    SG = FF.system['graph']
    a, b, c, alpha, beta, gamma = system['box']
    lx = np.round(a, 8)
    xy = np.round(b * np.cos(math.radians(gamma)), 8)
    xz = np.round(c * np.cos(math.radians(beta)), 8)
    ly = np.round(np.sqrt(b**2 - xy**2), 8)
    yz = np.round((b * c * np.cos(math.radians(alpha)) - xy * xz) / ly, 8)
    lz = np.round(np.sqrt(c**2 - xz**2 - yz**2), 8)

    preffix = cifname.split('/')[-1].split('.')[0]
    name = preffix + '.gin'

    with open(outdir + os.sep + name, 'w') as gin:

        if noautobond:
            gin.write('opti conp free zsisa noautobond cartesian\n')
        else:
            gin.write('opti conp free zsisa cartesian\n')
        gin.write('vectors\n')
        gin.write(str(lx) + ' 0.0 0.0' '\n')
        gin.write('0.0 ' + str(ly) + ' 0.0' + '\n')
        gin.write('0.0 0.0 ' + str(lz) + '\n')
        gin.write('cartesian\n')

        for a in SG.nodes(data=True):
            atom_data = a[1]
            # index = atom_data['index']
            force_field_type = atom_data['force_field_type']
            gulp_type = FF.atom_types[force_field_type]
            elem = FF.atom_element_symbols[force_field_type]
            pos = [np.round(v, 8) for v in atom_data['cartesian_position']]
            line = [elem + str(gulp_type), 'core', pos[0], pos[1], pos[2]]
            gin.write('{:5} {:<6} {:12.5f} {:12.5f} {:12.5f}'.format(*line))
            gin.write('\n')
        gin.write('\n')

        bonds = [(b, ty) for ty in FF.bond_data['all_bonds']
                 for b in FF.bond_data['all_bonds'][ty]]
        bonds.sort(key=lambda x: x[0][0])

        if noautobond:
            for bond in bonds:
                b, ty = bond
                comments = FF.bond_data['comments'][ty]
                bond_order = float(comments[-1].split('=')[-1])
                gulp_bond = gulp_bond_types[bond_order]
                line = ['connect', b[0], b[1], gulp_bond]
                gin.write('{:10} {:5} {:5} {:>10}'.format(*line))
                gin.write('\n')
            gin.write('\n')
        else:
            pass

        gin.write('species\n')
        for fft in FF.atom_types:

            elem = FF.atom_element_symbols[fft]
            gulp_type = FF.atom_types[fft]

            write_type = fft
            if fft == 'O_2_M':
                write_type = 'O_2'
            if fft == 'O_3_M':
                write_type = 'O_3'

            line = [elem + str(gulp_type), write_type]
            gin.write('{:4} {:7}'.format(*line))
            gin.write('\n')
        gin.write('\n')

        gin.write('library uff4mof.lib\n')
        gin.write('uff_bondorder custom 0.001\n')
        gin.write('output cif ' + 'OPT_' + preffix + '.cif' + '\n')
        gin.write('rspeed 0.1000\n')
        gin.write('temperature 300\n')

# GULP_inputs(['unopt_cifs/pcu_v1-6c_Zn_1_Ch_1B_2thiophene_Ch.cif', UFF4MOF, 'GULP_inputs', False, '1x1x1'])
