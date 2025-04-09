"""Run computations backed by MACE"""

from dataclasses import dataclass
from functools import lru_cache
from string import Template
from pathlib import Path
import os
from subprocess import run

import ase
from ase import units, io
from ase.io.lammpsrun import read_lammps_dump_text
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators import mace_mp
from ase.filters import UnitCellFilter
from ase.io import Trajectory
from ase.optimize import LBFGS

from mofa.model import MOFRecord
from mofa.simulation.interfaces import MDInterface
from mofa.utils.conversions import read_from_string

_mace_options = {
    "default": {
        "model": "medium",  # Can be 'small', 'medium', or 'large'
        "default_dtype": "float32",
        "dispersion": False,  # Whether to include dispersion corrections
    }
}

template_input = Template("""
units           metal
atom_style      atomic
atom_modify     map yes
newton          on
boundary        p p p


box             tilt large
read_data       data.lmp


pair_style mace no_domain_decomposition
pair_coeff * * $model_path $elements

# simulation

timestep            0.0005
fix                 fxnpt all npt temp 300.0 300.0 $$(200.0*dt) tri 1.0 1.0 $$(800.0*dt)
variable            Nevery equal $write_freq

thermo              10
thermo_style        custom step cpu dt time temp press pe ke etotal density xlo ylo zlo cella cellb cellc cellalpha cellbeta cellgamma
thermo_modify       flush yes

minimize            0. 1.0e-2 $min_steps 10000
reset_timestep      0

velocity            all create 300.0 12345

thermo              $${Nevery}

dump                trajectAll all custom $${Nevery} dump.lammpstrj.all id type element x y z
dump_modify         trajectAll element $elements

run                 $timesteps
undump              trajectAll
write_restart       relaxing.*.restart
""")


@lru_cache(1)
def load_model(device: str, level: str = 'default'):
    """Load a MACE calculator and cache it in device memory

    Args:
        device: Which device on which to load MACE
        level: What level of MACE to use
    """
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        import oneccl_bindings_for_pytorch as torch_ccl  # noqa: F401
    except ImportError:
        pass

    if level not in _mace_options:
        raise ValueError(f"No presets for {level}")
    options = _mace_options[level]
    return mace_mp(device=device, **options)


def _load_structure(mof: MOFRecord, structure_source: tuple[str, int] | None):
    """Read the appropriate input structure"""
    if structure_source is None:
        return mof.atoms
    else:
        traj, ind = structure_source
        return read_from_string(mof.md_trajectory[traj][ind][-1], "vasp")


@dataclass
class MACERunner(MDInterface):
    """Interface for running pre-defined MACE workflows"""

    lammps_cmd: list[str] | None = None
    """Command used to invoke MACE"""
    model_path: Path = None
    """Path to the LAMMPS-compatible model file. Ignored if using ASE"""
    run_dir: Path = Path("mace-runs")
    """Path in which to store MACE computation files"""
    traj_name = 'mace_mp'
    md_supercell: int = 2
    """How large of a supercell to use for the MD calculation"""
    device: str = 'cpu'
    """Which device to use for the calculations"""

    def run_single_point(
            self,
            mof: MOFRecord,
            level: str = "default",
            structure_source: tuple[str, int] | None = None,
    ) -> tuple[ase.Atoms, Path]:
        """Perform a single-point computation at a certain level

        Args:
            mof: Structure to be run
            level: Name of the level of computation to perform
            structure_source: Name of the MD trajectory and frame ID from which to source the
                input structure. Default is to use the as-assembled structure
        Returns:
            - Structure with computed properties
            - Path to the run directory
        """
        atoms = _load_structure(mof, structure_source)
        return self._run_mace(mof.name, atoms, "single", level)

    def run_optimization(
            self,
            mof: MOFRecord,
            level: str = "default",
            structure_source: tuple[str, int] | None = None,
            steps: int = 8,
            fmax: float = 1e-2,
    ) -> tuple[ase.Atoms, Path]:
        """Perform a geometry optimization computation

        Args:
            mof: Structure to be run
            level: Name of the level of computation to perform
            structure_source: Name of the MD trajectory and frame ID from which to source the
                input structure. Default is to use the as-assembled structure
            steps: Maximum number of optimization steps
            fmax: Convergence threshold for optimization
        Returns:
            - Relaxed structure
            - Path to the run directory
        """
        atoms = _load_structure(mof, structure_source)
        return self._run_mace(mof.name, atoms, "optimize", level, steps, fmax)

    def _run_mace(
            self,
            name: str,
            atoms: ase.Atoms,
            action: str,
            level: str,
            steps: int = 8,
            fmax: float = 1e-2,
            loginterval: int = 1,
    ) -> tuple[ase.Atoms, Path]:
        """Run MACE in a special directory

        Args:
            name: Name used for the start of the directory
            atoms: Starting structure to use
            action: Which action to perform (single, optimize)
            level: Level of accuracy to use
            steps: Number of steps to run
            fmax: Convergence threshold for optimization
            loginterval: How often to log to a trajectory
        Returns:
            - Structure with computed properties
            - Absolute path to the run directory
        """
        # Create and move to output directory
        out_dir = self.run_dir / f"{name}-{action}-{level}"
        start_dir = Path().cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(out_dir)

        # Load the model and move it to the device
        calc = load_model(self.device, level)
        for model in calc.models:
            model.to(self.device)

        try:
            # Initialize MACE calculator
            atoms = atoms.copy()
            atoms.calc = calc

            # Run the calculation
            if action == "single":
                atoms.get_potential_energy()
            elif action == "optimize":
                ecf = UnitCellFilter(atoms, hydrostatic_strain=False)
                with Trajectory("relax.traj", mode="w") as traj:
                    dyn = LBFGS(ecf, logfile="relax.log", trajectory=traj)
                    dyn.run(fmax=fmax, steps=steps)
            elif action == "md":
                MaxwellBoltzmannDistribution(temperature_K=300, atoms=atoms)
                with Trajectory("md.traj", mode="w") as traj:
                    dyn = Inhomogeneous_NPTBerendsen(atoms,
                                                     # TODO (wardlt): Tweak these. Assumign a 10GPa bulk modulus as a low estimate
                                                     #  from https://pubs.rsc.org/en/content/articlehtml/2019/sc/c9sc04249k
                                                     timestep=0.5 * units.fs, temperature_K=300,
                                                     taut=500 * units.fs, pressure_au=0,
                                                     taup=1000 * units.fs, compressibility_au=4.57e-5 / units.bar,
                                                     trajectory=traj, logfile='npt.log', loginterval=loginterval)
                    dyn.run(steps=steps)
            else:
                raise ValueError(f"Action not supported: {action}")

            # Write the result to disk for easy retrieval
            atoms.write("atoms.extxyz")
        finally:
            os.chdir(start_dir)

            # Move the model back to the CPU
            for model in calc.models:
                model.to('cpu')

        # Remove the calculator from the atoms
        atoms.calc = None
        return atoms, out_dir.absolute()

    def run_md_with_lammps(
            self,
            name: str,
            atoms: ase.Atoms,
            min_steps: int,
            timesteps: int,
            write_freq: int
    ) -> list[tuple[int, ase.Atoms]]:
        # Make a run directory
        out_dir = self.run_dir / f"{name}-lammps"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write the input file
        elements = sorted(set(atoms.get_chemical_symbols()))
        inp_file = template_input.substitute(
            elements=" ".join(elements),
            model_path=str(self.model_path),
            write_freq=write_freq,
            min_steps=min_steps,
            timesteps=timesteps
        )
        inp_path = out_dir / 'in.lammps'
        inp_path.write_text(inp_file)

        # Write the structure
        data_path = out_dir / 'data.lmp'
        io.write(
            str(data_path), atoms, 'lammps-data',
            specorder=elements, bonds=False, masses=True
        )

        # Invoke LAMMPS
        with open(out_dir / 'stdout.lmp', 'w') as fp, open(out_dir / 'stderr.lmp', 'w') as fe:
            env = None
            proc = run(list(self.lammps_cmd) + ['-i', inp_path.name], cwd=out_dir, stdout=fp, stderr=fe, env=env)

        if proc.returncode != 0:
            raise ValueError(f'LAMMPS failed in {out_dir}')

        # Read the outputs
        with open(out_dir / 'dump.lammpstrj.all') as fp:
            return [(i * write_freq, strc) for i, strc in enumerate(read_lammps_dump_text(fp, slice(None)))]

    def run_molecular_dynamics(self,
                               mof: MOFRecord,
                               timesteps: int,
                               report_frequency: int) -> list[tuple[int, ase.Atoms]]:
        # Get the initial structure
        if self.traj_name in mof.md_trajectory:
            start_frame, strc = mof.md_trajectory[self.traj_name][-1]
            atoms = read_from_string(strc, 'vasp')
            continuation = True
        else:
            atoms = mof.atoms * ([self.md_supercell] * 3)
            start_frame = 0
            continuation = False

        # Run the MD trajectory
        if self.lammps_cmd is not None:
            output = self.run_md_with_lammps(
                name=mof.name,
                atoms=atoms,
                timesteps=timesteps - start_frame,
                min_steps=0 if continuation else 100,
                write_freq=report_frequency
            )
            if continuation:
                output.pop(0)

            # Increment the outputs by the start time
            return [(i + start_frame, atoms) for i, atoms in output]
        else:
            out_atoms, out_dir = self._run_mace(
                name=mof.name,
                level='default',
                action='md',
                atoms=atoms,
                steps=timesteps - start_frame,
                loginterval=report_frequency,
            )

            # Read in the trajectory file
            output = []
            for i, atoms in enumerate(io.iread(out_dir / 'md.traj')):
                if continuation and i == 0:
                    continue
                timestep = start_frame + report_frequency * i
                output.append((timestep, atoms))
            if output[-1][0] != timesteps:
                output.append((timesteps, out_atoms))
            return output
