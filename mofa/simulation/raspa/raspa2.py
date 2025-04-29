from dataclasses import dataclass
import subprocess
import os
from pathlib import Path
import shutil

import ase

from mofa.simulation.raspa import BaseRaspaRunner
from mofa.simulation.raspa.utils import calculate_cell_size

_file_dir = Path(__file__).parent / "files" / "raspa2_template"


@dataclass
class RASPA2Runner(BaseRaspaRunner):
    """Interface for running pre-defined RASPA2 workflows."""

    run_dir: Path = Path("raspa2-runs")
    """Path to store RASPA2 files"""

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
                    line = line.replace(
                        "UC_X UC_Y UC_Z", f"{uc_x} {uc_y} {uc_z}"
                    )
                if "CUTOFF" in line:
                    line = line.replace("CUTOFF", str(self.cutoff))
                if "CIFFILE" in line:
                    line = line.replace("CIFFILE", "input")
                f_out.write(line)

        shutil.move(
            f"{out_dir}/simulation.input.tmp", f"{out_dir}/simulation.input"
        )

    def _read_outputs(self, out_dir: Path, atoms: ase.Atoms, adsorbate: str) -> tuple[float, float, float, float]:
        system_dir = os.path.join(out_dir, "Output", "System_0")
        output_file = next(
            (
                f
                for f in os.listdir(system_dir)
                if f.startswith("output_") and f.endswith(".data")
            ),
            None,
        )
        if not output_file:
            raise FileNotFoundError(
                "No output_*.data file found in Output/System_0"
            )
        output_path = os.path.join(system_dir, output_file)
        mol_kg_line = mg_g_line = density_line = None
        with open(output_path, "r") as file:
            for line in file:
                if "Average loading absolute [mol/kg framework]" in line:
                    mol_kg_line = line.strip()
                elif (
                        "Average loading absolute [milligram/gram framework]"
                        in line
                ):
                    mg_g_line = line.strip()
                elif "Framework Density" in line:
                    density_line = line.strip()

        if not all([mol_kg_line, mg_g_line, density_line]):
            raise ValueError("One or more expected lines were not found.")

        uptake_mol_kg = float(mol_kg_line.split()[5])
        error_mol_kg = float(mol_kg_line.split()[7])

        density_kg_m3 = float(density_line.split()[2])
        uptake_mg_g = float(mg_g_line.split()[5])
        error_mg_g = float(mg_g_line.split()[7])

        # Unit conversion to g/L
        uptake_g_L = uptake_mg_g * density_kg_m3 / 1000
        error_g_L = error_mg_g * density_kg_m3 / 1000
        return uptake_mol_kg, error_mol_kg, uptake_g_L, error_g_L
