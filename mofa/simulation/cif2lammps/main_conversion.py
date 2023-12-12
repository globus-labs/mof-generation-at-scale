from __future__ import print_function
import multiprocessing
import argparse
from multiprocessing import Pool
import numpy as np
import glob
import os
import time
import logging
from .write_lammps_data import lammps_inputs
from .write_GULP_inputs import GULP_inputs

from .UFF4MOF_construction import UFF4MOF
from .UFF_construction import UFF
from .Dreiding_construction import Dreiding
from .zeoliteFFs_construction import MZHB
from .ZIFFF_construction import ZIFFF

# add more force field classes here as they are made

logger = logging.getLogger(__name__)


def single_conversion(
        cif,
        force_field=UFF4MOF,
        ff_string='UFF4MOF',
        small_molecule_force_field=None,
        outdir='unopt_lammps_data',
        charges=False,
        parallel=False,
        replication='1x1x1',
        read_cifs_pymatgen=False,
        add_molecule=None,
        small_molecule_file=None):
    logger.debug('converting ', cif, '...')
    lammps_inputs([cif,
                   force_field,
                   ff_string,
                   small_molecule_force_field,
                   outdir,
                   charges,
                   replication,
                   read_cifs_pymatgen,
                   add_molecule,
                   small_molecule_file])


def serial_conversion(
        directory,
        force_field=UFF4MOF,
        ff_string='UFF4MOF',
        small_molecule_force_field=None,
        outdir='unopt_lammps_data',
        charges=False,
        parallel=False,
        replication='1x1x1',
        read_cifs_pymatgen=False,
        add_molecule=None,
        small_molecule_file=None):
    try:
        os.mkdir(outdir)
    except OSError:
        pass

    logger.debug('conversion running serial on a single core')

    cifs = sorted(glob.glob(directory + os.sep + '*.cif'))
    for cif in cifs:
        logger.debug('converting ', cif, '...')
        lammps_inputs([cif,
                       force_field,
                       ff_string,
                       small_molecule_force_field,
                       outdir,
                       charges,
                       replication,
                       read_cifs_pymatgen,
                       add_molecule,
                       small_molecule_file])

    logger.debug('--- cifs in', directory, 'converted and placed in', outdir, '---')


def parallel_conversion(
        directory,
        force_field=UFF4MOF,
        ff_string='UFF4MOF',
        small_molecule_force_field=None,
        outdir='unopt_lammps_data',
        charges=False,
        parallel=True,
        replication='1x1x1',
        read_cifs_pymatgen=False,
        add_molecule=None,
        small_molecule_file=None):
    try:
        os.mkdir(outdir)
    except OSError:
        pass

    logger.debug('conversion running on ' + str(multiprocessing.cpu_count()) + ' cores')

    cifs = sorted(glob.glob(directory + os.sep + '*.cif'))
    args = [[cif,
             force_field,
             ff_string,
             small_molecule_force_field,
             outdir,
             charges,
             replication,
             read_cifs_pymatgen,
             add_molecule,
             small_molecule_file] for cif in cifs]
    pool = Pool(multiprocessing.cpu_count())
    pool.map_async(lammps_inputs, args)
    # results_par = pool.map_async(lammps_inputs, args)
    pool.close()
    pool.join()

    logger.debug('--- cifs in', directory, 'converted and placed in', outdir, '---')


def parallel_GULP_conversion(
        directory,
        force_field=UFF4MOF,
        outdir='GULP_inputs',
        charges=False,
        parallel=True,
        replication='1x1x1',
        GULP=True,
        noautobond=True):
    try:
        os.mkdir(outdir)
    except OSError:
        pass

    logger.debug('conversion running on ' + str(multiprocessing.cpu_count()) + ' cores')

    cifs = sorted(glob.glob(directory + os.sep + '*.cif'))
    args = [[cif, force_field, outdir, charges, replication, noautobond]
            for cif in cifs]
    pool = Pool(multiprocessing.cpu_count())
    pool.map_async(GULP_inputs, args)
    # results_par = pool.map_async(GULP_inputs, args)
    pool.close()
    pool.join()

    logger.debug('--- cifs in', directory, 'converted and placed in', outdir, '---')


def run_conversion():
    parser = argparse.ArgumentParser(
        description='Optional arguments for running cif2lammps')
    parser.add_argument(
        '--cifs',
        action='store',
        dest='directory',
        type=str,
        required=True,
        help='the cifs to convert')
    parser.add_argument(
        '--force_field',
        action='store',
        dest='ff_string',
        type=str,
        required=False,
        default='UFF4MOF',
        help='the force field to use')
    parser.add_argument(
        '--small_molecule_force_field',
        action='store',
        dest='sm_ff_string',
        type=str,
        required=False,
        default='TraPPE',
        help='the force field to use for small molecules')
    parser.add_argument(
        '--outdir',
        action='store',
        dest='outdir',
        type=str,
        required=False,
        default='unopt_lammps_data',
        help='where to write the lammps inputs')
    parser.add_argument(
        '--charges',
        action='store_true',
        dest='charges',
        required=False,
        default=False,
        help='switch on charges')
    parser.add_argument(
        '--replication',
        action='store',
        dest='replication',
        type=str,
        required=False,
        default='1x1x1',
        help='replications to use')
    parser.add_argument(
        '--GULP',
        action='store_true',
        dest='GULP',
        required=False,
        default=False,
        help='write GULP inputs instead of LAMMPS')
    parser.add_argument(
        '--parallel',
        action='store_true',
        dest='parallel',
        required=False,
        default=False,
        help='switch on parallel conversion')
    parser.add_argument(
        '--read_cifs_pymatgen',
        action='store_true',
        dest='read_cifs_pymatgen',
        required=False,
        default=False,
        help='use ASE to read CIF inputs')
    parser.add_argument(
        '--add_molecule',
        action='store',
        dest='add_molecule',
        required=False,
        default=None,
        help='name of a molecule to write molecule file for')
    parser.add_argument(
        '--small_molecule_file',
        action='store',
        dest='sm_file',
        type=str,
        required=False,
        default=None,
        help='a cif, xyz, or pdb of small molecules to be added, should be in cifs folder')

    args = parser.parse_args()
    logger.debug(args)

    if args.add_molecule is not None:
        # should be name of molecule, model, number to add
        add_molecule = args.add_molecule.split(', ')
        add_molecule[2] = int(add_molecule[2])
    else:
        add_molecule = args.add_molecule

    ff_dict = {
        'UFF4MOF': UFF4MOF,
        'UFF': UFF,
        'Dreiding': Dreiding,
        'MZHB': MZHB,
        'ZIFFF': ZIFFF}
    force_field = ff_dict[args.ff_string]

    optional_arguments = {
        'force_field': force_field,
        'ff_string': args.ff_string,
        'small_molecule_force_field': args.sm_ff_string,
        'outdir': args.outdir,
        'charges': args.charges,
        'replication': args.replication,
        'read_cifs_pymatgen': args.read_cifs_pymatgen,
        'add_molecule': add_molecule,
        'small_molecule_file': args.sm_file}

    if args.GULP:
        logger.debug('converting to GULP format...')
        parallel_GULP_conversion(args.directory, **optional_arguments)

    if args.parallel:
        parallel_conversion(args.directory, **optional_arguments)
    else:
        serial_conversion(args.directory, **optional_arguments)


if __name__ == '__main__':
    start_time = time.time()
    run_conversion()
    logger.debug("conversion took %s seconds " %
                 np.round((time.time() - start_time), 3))
