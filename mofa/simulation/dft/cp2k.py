"""Run computations backed by CP2K"""
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Optional
from pathlib import Path
import shutil
import uuid
import time
import os

from ase.calculators.calculator import Calculator
from ase.calculators.cp2k import CP2K
from ase import units

from mofa.simulation.dft.base import BaseDFTRunner

_file_dir = Path(__file__).parent.joinpath('files').absolute()

_cp2k_options = {
    'default': {
        'basis_set': 'DZVP-MOLOPT-SR-GTH',
        'basis_set_file': "BASIS_MOLOPT",
        'pseudo_potential': "GTH-PBE",
        'potential_file': "GTH_POTENTIALS",
        'xc': None,
        'cutoff': 600 * units.Ry
    }
}


@dataclass
class CP2KRunner(BaseDFTRunner):
    """Interface for running pre-defined CP2K workflows"""

    dft_cmd: str = 'cp2k_shell'
    run_dir: Path = Path('cp2k-runs')
    close_cp2k: bool = False
    """Whether to close CP2K after a successful job"""
    _calc: Optional[CP2K] = field(default=None, init=False, repr=False)
    """Holds the calculator"""
    ignore_failure: bool = True
    """Whether to ignore convergence failures"""

    @contextmanager
    def _make_calc(self, level: str, out_dir: Path) -> Calculator:

        # Get the template for this level of computation
        template_file = _file_dir / f'cp2k-{level}-template.inp'
        if not template_file.is_file():
            raise ValueError(f'Template not found for {level}')
        inp = template_file.read_text()
        if self.ignore_failure:
            # Does not work with CP2K<2024.1, which is what we're running on Polaris but not what comes with Ubuntu
            inp = inp.replace("&SCF\n", "&SCF\n         IGNORE_CONVERGENCE_FAILURE\n")

        # Get the other settings
        if level not in _cp2k_options:
            raise ValueError(f'No presents for {level}')
        options = _cp2k_options[level]

        # Check the run-directory
        if self.close_cp2k:
            run_dir = out_dir
        elif self._calc is None:
            run_dir = self.run_dir / f'temp-{uuid.uuid4()}'
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = self._calc.run_dir

        start_dir = Path.cwd()
        os.chdir(run_dir)
        try:
            if self._calc is None:
                self._calc = CP2K(
                    command=self.dft_cmd,
                    directory=".",
                    inp=inp,
                    max_scf=128,
                    **options,
                )
                self._calc.run_dir = run_dir
            yield self._calc  # Pass the calculator to be used externally
        except AssertionError:
            time.sleep(30)  # Give time for CP2K to exit cleanly
            self._calc = None
            raise
        finally:
            os.chdir(start_dir)  # Paths are relative to the start directory

        if self.close_cp2k:
            self._calc = None
        else:
            # Copy files from the run directory to here
            for path in self._calc.run_dir.iterdir():
                shutil.copy(path, out_dir / path.name)
