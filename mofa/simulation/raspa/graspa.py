"""Interface to the CUDA Version of RASPA, `gRASPA <https://github.com/snurr-group/gRASPA>`_"""
from dataclasses import dataclass
from pathlib import Path
import subprocess
import shutil

import ase

from .base import BaseRaspaRunner
from .utils import calculate_cell_size

_file_dir = Path(__file__).parent / "files" / "graspa_template"


@dataclass
class gRASPARunner(BaseRaspaRunner):
    """Interface for running pre-defined gRASPA workflows."""

    graspa_command: list[str] = ("graspa.x",)
    """Invocation used to run gRASPA on this system"""
    run_dir: Path = Path("graspa-runs")
    """Path to store gRASPA files"""

    def _write_inputs(self,
                      out_dir: Path,
                      atoms: ase.Atoms,
                      cycles: int,
                      adsorbate: str,
                      temperature: float,
                      pressure: float):
        # Copy other input files (simulation.input, force fields and definition files) from template folder.
        subprocess.run(f"cp {_file_dir}/* {out_dir}/", shell=True)
        [uc_x, uc_y, uc_z] = calculate_cell_size(atoms=atoms)

        # Modify input from template simulation.input
        with (
            open(f"{out_dir}/simulation.input", "r") as f_in,
            open(f"{out_dir}/simulation.input.tmp", "w") as f_out,
        ):
            for line in f_in:
                if "NCYCLE" in line:
                    line = line.replace("NCYCLE", str(cycles))
                if "ADSORBATE" in line:
                    line = line.replace("ADSORBATE", adsorbate)
                if "TEMPERATURE" in line:
                    line = line.replace("TEMPERATURE", str(temperature))
                if "PRESSURE" in line:
                    line = line.replace("PRESSURE", str(pressure))
                if "UC_X UC_Y UC_Z" in line:
                    line = line.replace("UC_X UC_Y UC_Z", f"{uc_x} {uc_y} {uc_z}")
                if "CUTOFF" in line:
                    line = line.replace("CUTOFF", str(self.cutoff))
                if "CIF" in line:
                    line = line.replace("CIF", "input")
                f_out.write(line)

        shutil.move(f"{out_dir}/simulation.input.tmp", f"{out_dir}/simulation.input")

    def _read_outputs(self, out_dir: Path, atoms: ase.Atoms, adsorbate: str) -> tuple[float, float, float, float]:
        # Get output from raspa.log file
        results = []
        with open(f"{out_dir}/raspa.err", "r") as rf:
            for line in rf:
                if "Overall: Average" in line:
                    results.append(line.strip())

        result_mol_kg = results[-5].split(",")
        uptake_mol_kg = result_mol_kg[0].split()[-1]
        error_mol_kg = result_mol_kg[1].split()[-1]

        result_g_L = results[-3].split(",")
        uptake_g_L = result_g_L[0].split()[-1]
        error_g_L = result_g_L[1].split()[-1]
        return float(uptake_mol_kg), float(error_mol_kg), float(uptake_g_L), float(error_g_L)
