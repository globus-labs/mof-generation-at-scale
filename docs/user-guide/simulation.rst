Simulation Runners
==================

MOFA employs "runners" which invoke different scientific software
through interfaces appropriate for workflow engines.

Use any runner by first creating a runner object with any
settings specific a certain supercomputer, then call
the methods of that object:

.. code-block:: python

    runner = ExampleRunner(example_cmd='/path/to/example')
    output = runner.do_science(mof_strc, iterations)

The functions available are determined by the type of runner.

.. contents:: Available Runners
    :depth: 2

MD Runners
----------

MOFA uses constant-pressure, constant-temperature (NPT) molecular dyanmics 
to evaluate whether MOFs are dynamically stable.
We assume that MOFs which change in volume signficiantly over timescales accessible to MD (<<1 ms)
are unlikely to be stable in real-world conditions.

The primary tool for running molecular dynamics is the :class:`~mofa.simulation.mace.MACERunner`,
which uses LAMMPS to run molecular dynamics with forces from an
`MACE interatomic potential <https://mace-docs.readthedocs.io/en/latest>`_.

The input to the function is a :class:`~mofa.models.MOFRecord` object, how many timesteps to run,
and how often to save a structure.
The runner will determine the starting point based on the latest-available timestep in the record,
run molecular dynamics with LAMMPS, then return snapshots at the requesed intervals.

DFT Runners
-----------

Density Functional Theory (DFT) runners compute the energy of a MOF
along with the forces and partial charges of each atom from quantum mechanics.
MOFA currently supports DFT implemented using `CP2K <https://www.cp2k.org/>`_
and `PWDFT <https://github.com/ebylaska/PWDFT>`_

Each runner is based on the :class:`mofa.simulation.dft.base.BaseDFTRunner` class,
which has two functions: one to evaluate a structure as-provided,
and a second to relax the structure before evaluating.

The output of the functions are an ASE Atoms object holding the energies and forces,
and a path to the output directory.
The output directory contains at least a file named ``valence.cube`` holding the
charge density of the system, which are used to compute partial charges
with :meth:`~mofa.simulation.dft.compute_partial_charges`.
