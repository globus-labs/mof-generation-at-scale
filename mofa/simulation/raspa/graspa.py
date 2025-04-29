"""Interface to the CUDA Version of RASPA, `gRASPA <https://github.com/snurr-group/gRASPA>`_"""
from dataclasses import dataclass
from pathlib import Path
import subprocess
import shutil

from .base import BaseRaspaRunner
from .utils import get_cif_from_chargemol, write_cif, calculate_cell_size

_file_dir = Path(__file__).parent / "files" / "graspa_template"


@dataclass
class gRASPARunner(BaseRaspaRunner):
    """Interface for running pre-defined gRASPA workflows."""

    graspa_command: list[str] = ("graspa.x",)
    """Invocation used to run gRASPA on this system"""
    run_dir: Path = Path("graspa-runs")
    """Path to store gRASPA files"""

    def run_gcmc(
            self,
            name: str,
            cp2k_dir: Path,
            adsorbate: str,
            steps: int,
            temperature: float,
            pressure: float,
    ) -> tuple[float, float, float, float]:
        out_dir = self.run_dir / f"{name}_{adsorbate}_{temperature}_{pressure:0e}"
        out_dir.mkdir(parents=True, exist_ok=True)

        atoms = get_cif_from_chargemol(cp2k_dir)
        write_cif(atoms, out_dir=out_dir, name=name + ".cif")

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
                    line = line.replace("NCYCLE", str(steps))
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
                    line = line.replace("CIF", name)
                f_out.write(line)

        shutil.move(f"{out_dir}/simulation.input.tmp", f"{out_dir}/simulation.input")

        # Run gRASPA
        err_path = out_dir / 'raspa.err'
        with open(out_dir / 'raspa.log', 'w') as fp, open(err_path, 'w') as fe:
            result = subprocess.run(self.graspa_command, cwd=out_dir, stdout=fe, stderr=fp)

        if result.returncode != 0:
            raise ValueError(f'gRASPA failed: {err_path.read_text()[:128]}')

        # Get output from raspa.log file
        results = []
        with open(f"{out_dir}/raspa.log", "r") as rf:
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
