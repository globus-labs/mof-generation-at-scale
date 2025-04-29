"""Interface to a branch of gRASPA that uses SYCL

Source: https://github.com/abagusetty/gRASPA/tree/sync-launches
"""
from dataclasses import dataclass
import subprocess
from pathlib import Path
import shutil

from mofa.simulation.raspa import BaseRaspaRunner
from mofa.simulation.raspa.utils import write_cif, calculate_cell_size, get_cif_from_chargemol

_file_dir = Path(__file__).parent / "files" / "graspa_sycl_template"


@dataclass
class GRASPASyclRunner(BaseRaspaRunner):
    """Interface for running pre-defined gRASPA-sycl workflows."""

    graspa_command: list[str] = ()
    """Invocation used to run gRASPA-sycl on this system"""

    run_dir: Path = Path("gRASPA-sycl-runs")
    """Path to store gRASPA-sycl files"""
    delete_finished: bool = False
    """Whether to remove the directory after run"""

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
                if "CIFFILE" in line:
                    line = line.replace("CIFFILE", name)
                f_out.write(line)

        shutil.move(f"{out_dir}/simulation.input.tmp", f"{out_dir}/simulation.input")

        # Run gRASPA-sycl
        with open(out_dir / "raspa.log", "w") as fp, open(out_dir / "raspa.err", "w") as fe:
            result = subprocess.run(self.graspa_command, cwd=out_dir, stdout=fp, stderr=fe)

        if result.returncode != 0:
            raise ValueError(f'RASPA failed in {out_dir}')

        # Get output from Output/ folder
        with open(f"{out_dir}/raspa.log", "r") as rf:
            for line in rf:
                if "UnitCells" in line:
                    unitcell_line = line.strip()
                elif "Overall: Average:" in line:
                    uptake_line = line.strip()

        # gRASPA-sycl only output the total number of molecules
        # This section is for unit conversion.
        unitcell = unitcell_line.split()[4:]
        unitcell = [int(float(i)) for i in unitcell]
        uptake_total_molecule = float(uptake_line.split()[2][:-1])
        error_total_molecule = float(uptake_line.split()[4][:-1])

        # Get unit in mol/kg
        framework_mass = sum(atoms.get_masses())
        framework_mass = framework_mass * unitcell[0] * unitcell[1] * unitcell[2]
        uptake_mol_kg = uptake_total_molecule / framework_mass * 1000
        error_mol_kg = error_total_molecule / framework_mass * 1000

        # Get unit in g/L
        framework_vol = atoms.get_volume()  # in Angstrom^3
        framework_vol_in_L = framework_vol * 1e-27 * unitcell[0] * unitcell[1] * unitcell[2]

        # Hard code for CO2 and H2
        if adsorbate == "CO2":
            molar_mass = 44.0098
        elif adsorbate == "H2":
            molar_mass = 2.02
        else:
            raise ValueError(f"Adsorbate {adsorbate} is not supported.")
        uptake_g_L = uptake_total_molecule / (6.022 * 1e23) * molar_mass / framework_vol_in_L
        error_g_L = error_total_molecule / (6.022 * 1e23) * molar_mass / framework_vol_in_L

        # Remove gRASPA simulation directory
        if self.delete_finished:
            shutil.rmtree(out_dir)

        return uptake_mol_kg, error_mol_kg, uptake_g_L, error_g_L
