"""Run computations backed by CP2K"""
from dataclasses import dataclass
from pathlib import Path
import os

from ase.calculators.cp2k import CP2K
from ase import units

from mofa.model import MOFRecord

_file_dir = Path(__file__).parent / 'files'

_cp2k_options = {
    'pbe': {
        'basis_set': 'DZVP-MOLOPT-SR-GTH',
        'basis_set_file': "BASIS_MOLOPT",
        'pseudo_potential': "GTH-PBE",
        'potential_file': "GTH_POTENTIALS",
        'cutoff': 600 * units.Ry
    }
}


@dataclass
class CP2KRunner:
    """Interface for running pre-defined CP2K workflows"""

    cp2k_invocation: list[str] = ('cp2k_shell',)
    """Invocation used to run CP2K on this system"""

    run_dir: Path = Path('cp2k-runs')
    """Path in which to store CP2K files"""

    run_ddec: bool = False
    """whether to run DDEC after CP2k run or not"""

    def run_single_point(self, mof: MOFRecord, level: str = 'pbe') -> Path:
        """Perform a single-point computation at a certain level

        Args:
            mof: Structure to be run
            level: Name of the level of DFT computation to perform
        Returns:
            Path to the output files
        """

        # Get the template for this level of computation
        template_file = _file_dir / f'cp2k-{level}-template.inp'
        if not template_file.is_file():
            raise ValueError(f'Template not found for {level}')

        # Get the other settings
        if level not in _cp2k_options:
            raise ValueError(f'No presents for {level}')
        options = _cp2k_options[level]

        # Open then move to the output directory
        # CP2K does not like long directory names in input files, so we move to the local directory
        out_dir = self.run_dir / f'{mof.name}-single-{level}'
        start_dir = Path().cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'cp2k.out').write_text('')  # Clear old content
        os.chdir(out_dir)
        try:
            with CP2K(
                    command=self.cp2k_invocation,
                    directory=".",
                    inp=template_file.read_text(),
                    **options,
            ) as calc:

                # Run the calculation
                atoms = mof.atoms
                atoms.calc = calc
                atoms.get_potential_energy()

                # Write the
                atoms.write('atoms.json')

            if self.run_ddec:
                chargemol_nompt = 2  # OpenMP threads number for chargemol
                # run ddec here
                import sys
                atomic_density_folder_path = str(Path(sys.prefix) / "share" / "chargemol" / "atomic_densities")
                if not atomic_density_folder_path.endswith("/"):
                    atomic_density_folder_path = atomic_density_folder_path + "/"
                write_str = None
                with open(_file_dir / "chargemol" / "job_control.txt", "r") as rf:
                    write_str = rf.read().replace("$$$ATOMIC_DENSITY_DIR$$$", atomic_density_folder_path)
                with open("job_control.txt", "w") as wf:
                    wf.write(write_str)
                cube_fname = [x for x in os.listdir() if x.endswith(".cube")][-1]
                # my local CP2k is not renaming the output automatically, so an extra renaming step is added here
                os.rename(cube_fname, "valence_density.cube")
                # run chargemol in the conda env bin
                # chargemol uses all cores for OpenMP parallelism if no OMP_NUM_THREADS is set
                os.system(f"OMP_NUM_THREADS={chargemol_nompt} chargemol")
                os.chdir(start_dir)
        finally:
            os.chdir(start_dir)
        return out_dir.absolute()
