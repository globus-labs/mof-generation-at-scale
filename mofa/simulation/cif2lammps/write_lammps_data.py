from .cif2system import initialize_system, replication_determination
from .small_molecule_construction import add_small_molecules, include_molecule_file, read_small_molecule_file
from . import atomic_data
import logging
import os
import numpy as np
import datetime
import math
import warnings
import networkx as nx
from itertools import combinations

from . import UFF4MOF_constants
from . import UFF_constants
from . import Dreiding_constants

logger = logging.getLogger(__name__)

# add more force field classes here as they are made

UFF4MOF_atom_parameters = UFF4MOF_constants.UFF4MOF_atom_parameters
UFF4MOF_bond_orders_0 = UFF4MOF_constants.UFF4MOF_bond_orders_0

UFF_atom_parameters = UFF_constants.UFF_atom_parameters
UFF_bond_orders_0 = UFF_constants.UFF_bond_orders_0

Dreiding_atom_parameters = Dreiding_constants.Dreiding_atom_parameters
Dreiding_bond_orders_0 = Dreiding_constants.Dreiding_bond_orders_0

mass_key = atomic_data.mass_key


def isfloat(value):
    """
        determines if a value is a float
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def lammps_inputs(args):

    cifname, force_field, ff_string, sm_ff_string, outdir, charges, replication, read_pymatgen, add_molecule, sm_file = args

    # add more forcefields here as they are created
    if ff_string == 'UFF4MOF':
        FF_args = {
            'FF_parameters': UFF4MOF_atom_parameters,
            'bond_orders': UFF4MOF_bond_orders_0}
        cutoff = 12.50
        mixing_rules = 'shift yes mix geometric'
    elif ff_string == 'UFF':
        FF_args = {
            'FF_parameters': UFF_atom_parameters,
            'bond_orders': UFF_bond_orders_0}
        cutoff = 12.50
        mixing_rules = 'shift yes mix geometric'
    elif ff_string == 'Dreiding':
        FF_args = {
            'FF_parameters': Dreiding_atom_parameters,
            'bond_orders': Dreiding_bond_orders_0}
        cutoff = 12.50
        mixing_rules = 'shift yes mix arithmetic'
    elif ff_string == 'MZHB':
        FF_args = {'FF_parameters': None, 'bond_orders': None}
        cutoff = 12.50
        mixing_rules = 'shift yes mix arithmetic'
    elif ff_string == 'ZIFFF':
        FF_args = {'FF_parameters': 'gaff2', 'bond_orders': None}
        cutoff = 12.50
        mixing_rules = 'shift yes mix arithmetic'

    system = initialize_system(
        cifname,
        charges=charges,
        read_pymatgen=read_pymatgen)

    if sm_file is not None:
        direc = cifname.split(os.sep)[0]
        read_small_molecule_file(direc + os.sep + sm_file, system)

    system, replication = replication_determination(
        system, replication, cutoff)

    logger.debug('system initialized...')

    FF = force_field(system, cutoff, FF_args)
    FF.compile_force_field(charges=charges)

    if sm_ff_string is not None:
        add_small_molecules(FF, sm_ff_string)
    else:
        if len(FF.system['SM_graph'].nodes()) != 0:
            warnings.warn(
                'extra-framework molecules detected, but no small molecule force field is specified!')

    # write_cif_from_system(system, 'check.cif')
    first_line = "Created by Ryther's code on " + \
        str(datetime.datetime.now()) + ", modified by xyan11@uic.edu. "

    SG = FF.system['graph']
    N_atoms, ty_atoms = (len(SG.nodes()), len(FF.atom_types))
    N_bonds, ty_bonds = FF.bond_data['count']
    N_angles, ty_angles = FF.angle_data['count']

    try:
        N_dihedrals, ty_dihedrals = FF.dihedral_data['count']
    except AttributeError:
        N_dihedrals = 0
        ty_dihedrals = None

    try:
        N_impropers, ty_impropers = FF.improper_data['count']
    except AttributeError:
        N_impropers = 0
        ty_impropers = None

    if replication != '':
        suffix = ''.join(cifname.split(
            '/')[-1].split('.')[0:-1]) + '_' + replication
    else:
        suffix = ''.join(cifname.split('/')[-1].split('.')[0:-1])

    if sm_file is not None:

        NM = len(list(nx.connected_components(system['SM_graph'])))
        suffix += '_' + str(NM)

    maxIDs = (ty_atoms, ty_bonds, ty_angles, ty_dihedrals, ty_impropers)
    if add_molecule is not None:
        molfile, infile_add_lines, extra_types = include_molecule_file(
            FF, maxIDs, add_molecule)
        with open(outdir + os.sep + 'mol.' + suffix, 'w') as MF:
            MF.write(molfile)

    a, b, c, alpha, beta, gamma = system['box']
    cifbox = "# a, b, c, alpha, beta, gamma: " + "%.10f" % a + ", " + "%.10f" % b + ", " + "%.10f" % c + \
        ", " + "%.10f" % alpha + ", " + "%.10f" % beta + ", " + "%.10f" % gamma + " $$$atoms$$$"
    lx = np.round(a, 8)
    xy = np.round(b * np.cos(math.radians(gamma)), 8)
    xz = np.round(c * np.cos(math.radians(beta)), 8)
    ly = np.round(np.sqrt(b**2 - xy**2), 8)
    yz = np.round((b * c * np.cos(math.radians(alpha)) - xy * xz) / ly, 8)
    lz = np.round(np.sqrt(c**2 - xz**2 - yz**2), 8)

    data_name = 'data.' + suffix

    with open(outdir + os.sep + data_name, 'w') as data:
        data.write(first_line + '\n')
        data.write('\n')
        data.write('    ' + str(N_atoms) + ' atoms\n')
        data.write('    ' + str(N_bonds) + ' bonds\n')
        data.write('    ' + str(N_angles) + ' angles\n')
        data.write('    ' + str(N_dihedrals) + ' dihedrals\n')
        data.write('    ' + str(N_impropers) + ' impropers\n')
        data.write('\n')
        data.write('    ' + str(ty_atoms) + ' atom types\n')
        data.write('    ' + str(ty_bonds) + ' bond types\n')
        data.write('    ' + str(ty_angles) + ' angle types\n')
        data.write('    ' + str(ty_dihedrals) + ' dihedral types\n')
        data.write('    ' + str(ty_impropers) + ' improper types\n')
        data.write('\n')
        data.write('0.00000000 ' + str(lx) + ' xlo xhi\n')
        data.write('0.00000000 ' + str(ly) + ' ylo yhi\n')
        data.write('0.00000000 ' + str(lz) + ' zlo zhi\n')
        data.write(str(xy) + ' ' + str(xz) + ' ' + str(yz) + ' xy xz yz \n')
        data.write('\n')
        data.write('Masses \n')
        data.write('\n')
        for fft in FF.atom_masses:
            mass = FF.atom_masses[fft]
            aty = FF.atom_types[fft]
            sym = FF.atom_element_symbols[fft]
            data.write('{:>5} {:>10}  # {:>3}'.format(aty, mass, sym))
            data.write('\n')
        data.write('\n')

        pair_style = FF.pair_data['style']

        if 'hybrid' not in pair_style:

            data.write('Pair Coeffs\n')
            data.write('\n')

            for aty in FF.pair_data['params']:

                params = FF.pair_data['params'][aty]
                params = [np.round(x, 6) if isfloat(x) else x for x in params]
                comment = FF.pair_data['comments'][aty]
                style = FF.pair_data['style']

                # type
                data.write('    {:<3}'.format(aty))
                format_string = ' '.join(['{:{w}.{p}f}' if not np.issubdtype(
                    x, np.integer) else '{:{w}}' for x in params[1:]])
                data.write(format_string.format(*params[1:], w=12, p=5))
                data.write(' '.join(['  # '] + comment))
                data.write('\n')

        data.write('\n')
        data.write('Bond Coeffs\n')
        data.write('\n')

        for bty in FF.bond_data['params']:

            params = FF.bond_data['params'][bty]
            params = [np.round(x, 6) if isfloat(x) else x for x in params]
            comment = FF.bond_data['comments'][bty]
            style = FF.bond_data['style']

            if 'hybrid' in style:

                if params[0] == 'zero':
                    line = [bty, params[0]]
                    data.write('{:>5} {:>28}'.format(*line))
                    data.write('\n')
                    continue

                # type
                data.write('    {:<3}'.format(bty))
                # style needs to be written for hybrid
                data.write('{:<20}'.format(params[0]))

                try:
                    format_string = ' '.join(['{:{w}.{p}f}' if not np.issubdtype(
                        x, np.integer) else '{:{w}}' for x in params[1:]])
                except TypeError:
                    format_string = ' '.join(['{:{w}}' for x in params[1:]])

                data.write(format_string.format(*params[1:], w=12, p=5))
                data.write(' '.join(['  # '] + comment))
                data.write('\n')
            else:
                # type
                data.write('    {:<3}'.format(bty))

                try:
                    format_string = ' '.join(['{:{w}.{p}f}' if not np.issubdtype(
                        x, np.integer) else '{:{w}}' for x in params[1:]])
                except TypeError:
                    format_string = ' '.join(['{:{w}}' for x in params[1:]])

                data.write(format_string.format(*params[1:], w=12, p=5))
                data.write(' '.join(['  # '] + comment))
                data.write('\n')

        data.write('\n')
        data.write('Angle Coeffs\n')
        data.write('\n')

        for aty in FF.angle_data['params']:

            params = FF.angle_data['params'][aty]
            params = [np.round(x, 6) if isfloat(x) else x for x in params]
            comment = FF.angle_data['comments'][aty]
            style = FF.angle_data['style']

            if 'hybrid' in style:
                # type
                data.write('    {:<3}'.format(aty))
                # style needs to be written for hybrid
                data.write('{:<20}'.format(params[0]))

                try:
                    format_string = ' '.join(['{:{w}.{p}f}' if not np.issubdtype(
                        x, np.integer) else '{:{w}}' for x in params[1:]])
                except TypeError:
                    format_string = ' '.join(['{:{w}}' for x in params[1:]])

                data.write(format_string.format(*params[1:], w=12, p=5))
                data.write(' '.join(['  # '] + comment))
                data.write('\n')
            else:
                # type
                data.write('    {:<3}'.format(aty))

                try:
                    format_string = ' '.join(['{:{w}.{p}f}' if not np.issubdtype(
                        x, np.integer) else '{:{w}}' for x in params[1:]])
                except TypeError:
                    format_string = ' '.join(['{:{w}}' for x in params[1:]])

                data.write(format_string.format(*params[1:], w=12, p=5))
                data.write(' '.join(['  # '] + comment))
                data.write('\n')

        if N_dihedrals != 0:

            data.write('\n')
            data.write('Dihedral Coeffs\n')
            data.write('\n')

            for dty in FF.dihedral_data['params']:
                params = FF.dihedral_data['params'][dty]
                params = [np.round(x, 6) if isfloat(x) else x for x in params]
                comment = FF.dihedral_data['comments'][dty]
                style = FF.dihedral_data['style']

                if 'hybrid' in style:
                    # type
                    data.write('    {:<3}'.format(dty))
                    # style needs to be written for hybrid
                    data.write('{:<20}'.format(params[0]))
                    format_string = ' '.join(['{:{w}.{p}f}' if not np.issubdtype(
                        x, np.integer) else '{:{w}}' for x in params[1:]])
                    data.write(format_string.format(*params[1:], w=12, p=5))
                    data.write(' '.join(['  # '] + comment))
                    data.write('\n')
                else:
                    # type
                    data.write('    {:<3}'.format(dty))
                    format_string = ' '.join(['{:{w}.{p}f}' if not np.issubdtype(
                        x, np.integer) else '{:{w}}' for x in params[1:]])
                    data.write(format_string.format(*params[1:], w=12, p=5))
                    data.write(' '.join(['  # '] + comment))
                    data.write('\n')

        if N_impropers != 0:

            data.write('\n')
            data.write('Improper Coeffs\n')
            data.write('\n')

            for ity in FF.improper_data['params']:
                params = FF.improper_data['params'][ity]
                params = [np.round(x, 6) if isfloat(x) else x for x in params]
                comment = FF.improper_data['comments'][ity]
                style = FF.improper_data['style']

                if 'hybrid' in style:
                    # type
                    data.write('    {:<3}'.format(ity))
                    # style needs to be written for hybrid
                    data.write('{:<20}'.format(params[0]))
                    format_string = ' '.join(['{:{w}.{p}f}' if not np.issubdtype(
                        x, np.integer) else '{:{w}}' for x in params[1:]])
                    data.write(format_string.format(*params[1:], w=12, p=5))
                    data.write(' '.join(['  # '] + comment))
                    data.write('\n')
                else:
                    # type
                    data.write('    {:<3}'.format(ity))
                    format_string = ' '.join(['{:{w}.{p}f}' if not np.issubdtype(
                        x, np.integer) else '{:{w}}' for x in params[1:]])
                    data.write(format_string.format(*params[1:], w=12, p=5))
                    data.write(' '.join(['  # '] + comment))
                    data.write('\n')

        data.write('\n')
        data.write('Atoms ' + cifbox + '\n')
        data.write('\n')
        total_charge = 0.0

        atom_ids = []
        atom_cif_labels = []
        for a in SG.nodes(data=True):

            atom_data = a[1]
            index = a[0]
            atom_ids.append(index)
            force_field_type = atom_data['force_field_type']
            lammps_type = FF.atom_types[force_field_type]
            charge = atom_data['charge']
            cif_label = atom_data['cif_label']
            atom_cif_labels.append(cif_label)
            total_charge += charge
            pos = [np.round(v, 8) for v in atom_data['cartesian_position']]
            frac_pos = [np.round(v, 8)
                        for v in atom_data['fractional_position']]

            data.write(
                '{:>5} {:<5} {:<5} {:8.5f} {:12.5f} {:12.5f} {:12.5f}   # {:>5} {:12.8f} {:12.8f} {:12.8f}'.format(
                    index,
                    atom_data['mol_flag'],
                    lammps_type,
                    charge,
                    pos[0],
                    pos[1],
                    pos[2],
                    cif_label,
                    frac_pos[0],
                    frac_pos[1],
                    frac_pos[2],
                ))
            data.write('\n')

        id2label = dict(zip(atom_ids, atom_cif_labels))
        if charges and abs(total_charge) > 0.001:
            warnings.warn(
                'There is a potentially significant net charge of ' +
                str(total_charge))

        data.write('\n')
        data.write('Bonds\n')
        data.write('\n')

        bond_index = 0
        for bond_type in FF.bond_data['all_bonds']:
            for bond in FF.bond_data['all_bonds'][bond_type]:
                bond_index += 1
                data.write('{:>5} {:<5} {:<6} {:<6}   # {:>6} {:>6}'.format(bond_index,
                                                                            bond_type,
                                                                            bond[0],
                                                                            bond[1],
                                                                            id2label[bond[0]],
                                                                            id2label[bond[1]]))
                data.write('\n')

        data.write('\n')
        data.write('Angles\n')
        data.write('\n')

        angle_index = 0
        for angle_type in FF.angle_data['all_angles']:
            for angle in FF.angle_data['all_angles'][angle_type]:
                angle_index += 1
                data.write('{:>5} {:<5} {:<6} {:<6} {:<6}   # {:>6} {:>6} {:>6}'.format(angle_index,
                                                                                        angle_type,
                                                                                        angle[0],
                                                                                        angle[1],
                                                                                        angle[2],
                                                                                        id2label[angle[0]],
                                                                                        id2label[angle[1]],
                                                                                        id2label[angle[2]]))
                data.write('\n')

        if N_dihedrals != 0:

            data.write('\n')
            data.write('Dihedrals\n')
            data.write('\n')

            dihedral_index = 0
            for dihedral_type in FF.dihedral_data['all_dihedrals']:
                for dihedral in FF.dihedral_data['all_dihedrals'][dihedral_type]:
                    dihedral_index += 1
                    data.write('{:>5} {:<5} {:<6} {:<6} {:<6} {:<6}   # {:>6} {:>6} {:>6} {:>6}'.format(dihedral_index,
                                                                                                        dihedral_type,
                                                                                                        dihedral[0],
                                                                                                        dihedral[1],
                                                                                                        dihedral[2],
                                                                                                        dihedral[3],
                                                                                                        id2label[dihedral[0]],
                                                                                                        id2label[dihedral[1]],
                                                                                                        id2label[dihedral[2]],
                                                                                                        id2label[dihedral[3]]))
                    data.write('\n')

        if N_impropers != 0:

            data.write('\n')
            data.write('Impropers\n')
            data.write('\n')

            improper_index = 0
            for improper_type in FF.improper_data['all_impropers']:
                for improper in FF.improper_data['all_impropers'][improper_type]:
                    improper_index += 1
                    data.write('{:>5} {:<5} {:<6} {:<6} {:<6} {:<6}   # {:>6} {:>6} {:>6} {:>6}'.format(improper_index,
                                                                                                        improper_type,
                                                                                                        improper[0],
                                                                                                        improper[1],
                                                                                                        improper[2],
                                                                                                        improper[3],
                                                                                                        id2label[improper[0]],
                                                                                                        id2label[improper[1]],
                                                                                                        id2label[improper[2]],
                                                                                                        id2label[improper[3]]))
                    data.write('\n')

    with open(outdir + os.sep + 'in.' + suffix, 'w') as infile:

        infile.write('units           real\n')
        infile.write('atom_style      full\n')
        infile.write('boundary        p p p\n')
        infile.write('\n')

        infile.write('bond_style      ' + FF.bond_data['style'] + '\n')
        infile.write('angle_style     ' + FF.angle_data['style'] + '\n')

        try:
            infile.write('dihedral_style  ' + FF.dihedral_data['style'] + '\n')
        except AttributeError:
            pass
        try:
            infile.write('improper_style  ' + FF.improper_data['style'] + '\n')
        except AttributeError:
            pass

        infile.write('\n')
        infile.write('dielectric      1.0\n')

        read_data_append_string = ''
        if add_molecule is not None:
            read_data_append_string = ' '
            et_strings = [
                'extra/atom/types',
                'extra/bond/types',
                'extra/angle/types',
                'extra/dihedral/types',
                'extra/improper/types']
            for st, et in zip(et_strings, extra_types):
                if et is not None:
                    read_data_append_string += st + ' ' + str(et) + ' '

        if add_molecule is not None:

            for line in infile_add_lines:
                infile.write(line + '\n')

            try:
                for bal in FF.bond_data['infile_add_lines']:
                    infile.write(bal + '\n')
            except KeyError:
                pass

            try:
                for aal in FF.angle_data['infile_add_lines']:
                    infile.write(aal + '\n')
            except KeyError:
                pass

            try:
                for dal in FF.dihedral_data['infile_add_lines']:
                    infile.write(dal + '\n')
            except KeyError:
                pass

            try:
                for ial in FF.improper_data['infile_add_lines']:
                    infile.write(ial + '\n')
            except KeyError:
                pass

            infile.write('\n')

        pair_style = FF.pair_data['style']

        if 'hybrid' not in pair_style:
            if 'tip4p' in pair_style:
                add_arg = ' ' + ' '.join([str(FF.pair_data['O_type']),
                                          str(FF.pair_data['H_type']),
                                          str(FF.pair_data['H2O_bond_type']),
                                          str(FF.pair_data['H2O_angle_type']),
                                          str(FF.pair_data['M_site_dist']),
                                          '12.5',
                                          '12.5'])
                FF.pair_data['style'] = FF.pair_data['style'] + add_arg
                infile.write('pair_style      ' + FF.pair_data['style'] + '\n')
            else:
                infile.write('pair_style      ' +
                             FF.pair_data['style'] +
                             ' ' +
                             str(FF.cutoff) +
                             '\n')
        else:

            style_list = FF.pair_data['style'].split()

            for pos, st in enumerate(style_list):
                if 'lj/cut/tip4p/' in st:
                    add_arg = ' ' + ' '.join([
                        str(FF.pair_data['O_type']), str(FF.pair_data['H_type']),
                        str(FF.pair_data['H2O_bond_type']), str(FF.pair_data['H2O_angle_type']),
                        str(FF.pair_data['M_site_dist']), '12.5', '12.5',
                    ])

                    style_list[pos] = style_list[pos] + add_arg

            style_string = 'hybrid '
            for st in style_list[1:]:
                if 'lj/cut/tip4p/' not in st:
                    style_string += st + ' ' + str(FF.cutoff) + ' '
                else:
                    style_string += st + ' '

            infile.write('pair_style      ' + style_string + '\n')

        if add_molecule is not None:
            if 'TIP' in add_molecule[1] and 'hybrid' not in FF.pair_data['style']:
                mixing_rules = 'tail yes mix geometric'
            elif 'TIP' in add_molecule[1] and 'hybrid' in FF.pair_data['style']:
                mixing_rules = 'tail yes'
        else:
            if sm_ff_string is not None:
                if 'TIP4P' in sm_ff_string and 'hybrid' not in FF.pair_data['style']:
                    mixing_rules = 'tail yes mix geometric'
                elif 'TIP4P' in sm_ff_string and 'hybrid' in FF.pair_data['style']:
                    mixing_rules = 'tail yes'

        sb = FF.pair_data['special_bonds']
        infile.write('pair_modify     ' + mixing_rules + '\n')
        infile.write('special_bonds   ' + sb + '\n')

        # use ewald summation for long range solver unless using pair_style
        # lj/cut/tip4p/long
        if 'long' in pair_style and 'tip4p' not in pair_style:
            infile.write('kspace_style    ewald 1.0e-5\n')
        elif 'tip4p/long' in pair_style:
            infile.write('kspace_style    pppm/tip4p 1.0e-5\n')

        if 'hybrid' in pair_style:

            infile.write('\n')

            for aty0 in FF.pair_data['params']:

                params0 = FF.pair_data['params'][aty0]

                if len(params0) > 3:
                    raise ValueError(
                        'pair_style hybrid is only supported for lj type vdw interactions (two numeric parameters).')

                params0 = [np.round(x, 6) if isfloat(x)
                           else x for x in params0]
                style0, eps0, sig0 = params0

                comment = '  # ' + ' ' + \
                    ' '.join(FF.pair_data['comments'][aty0])

                line = ['pair_coeff', aty0, aty0, style0, eps0, sig0, comment]
                infile.write(
                    '{:12} {:<3} {:<3} {:20} {:10.6f} {:10.6f} {:<20}'.format(
                        *line))
                infile.write('\n')

            infile.write('\n')

            for (aty0, aty1) in combinations(FF.pair_data['params'], 2):

                params0 = FF.pair_data['params'][aty0]
                params0 = [np.round(x, 6) if isfloat(x)
                           else x for x in params0]
                style0, eps0, sig0 = params0
                comment0 = FF.pair_data['comments'][aty0][0]

                params1 = FF.pair_data['params'][aty1]
                params1 = [np.round(x, 6) if isfloat(x)
                           else x for x in params1]
                style1, eps1, sig1 = params1
                comment1 = FF.pair_data['comments'][aty1][0]

                comments = '  # ' + ' ' + comment0 + ' ' + comment1

                # current logic is to use the longer style, this actually works well when using lj/cut for
                # framework atoms and lj/cut + charge interactions for framework/molecule and molecule/molecule
                # interactions. This is mostly what I use pair_style hybrid
                # for.
                style = style0 if len(style0) > len(style1) else style1

                if 'geometric' in mixing_rules:
                    pair_eps = np.round(np.sqrt(eps0 * eps1), 6)
                    pair_sig = np.round(np.sqrt(sig0 * sig1), 6)
                elif 'arithmetic' in mixing_rules:
                    pair_eps = np.round(np.sqrt(eps0 * eps1), 6)
                    pair_sig = np.round((sig0 + sig1) / 2.0, 6)
                else:
                    pair_eps = np.round(np.sqrt(eps0 * eps1), 6)
                    pair_sig = np.round(np.sqrt(sig0 * sig1), 6)

                line = [
                    'pair_coeff',
                    aty0,
                    aty1,
                    style,
                    pair_eps,
                    pair_sig,
                    comments]

                infile.write(
                    '{:12} {:<3} {:<3} {:20} {:10.6f} {:10.6f} {:<20}'.format(
                        *line))
                infile.write('\n')

            infile.write('\n')

        infile.write('box             tilt large\n')
        infile.write(
            'read_data       ' +
            data_name +
            read_data_append_string +
            '\n\n')

        if add_molecule is not None:
            if 'TIP4P' in add_molecule[1]:
                group_line = 'group           H2O type ' + \
                    str(FF.pair_data['O_type']) + ' ' + str(FF.pair_data['H_type']) + '\n'
                shake_line = 'fix             H2O_shake H2O shake 0.0001 50 0 b ' + \
                    str(FF.pair_data['H2O_bond_type']) + ' a ' + str(FF.pair_data['H2O_angle_type']) + ' mol H2O_mol\n'
                infile.write(group_line)
                infile.write(shake_line)
        else:
            if sm_ff_string is not None:
                if 'TIP4P' in sm_ff_string:

                    group_line = 'group           H2O type ' + \
                        str(FF.pair_data['O_type']) + ' ' + str(FF.pair_data['H_type']) + '\n'
                    shake_line = 'fix             H2O_shake H2O shake 0.0001 50 0 b ' + \
                        str(FF.pair_data['H2O_bond_type']) + ' a ' + \
                        str(FF.pair_data['H2O_angle_type']) + ' t ' + \
                        str(FF.pair_data['O_type']) + ' ' + \
                        str(FF.pair_data['H_type']) + '\n'
                    infile.write(group_line)
                    infile.write(shake_line)

        infile.write('\n')
