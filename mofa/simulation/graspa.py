from dataclasses import dataclass
import ase
from pathlib import Path
import numpy as np
import subprocess
import os
import platform

_file_dir = Path(__file__).parent / 'files' / 'graspa_template'

class gRASPARunner:
    graspa_command: str
    run_dir: Path = Path('graspa-runs')
    """Interface for running pre-defined gRASPA workflows"""

    def _calculate_cell_size(self, atoms: ase.Atoms, cutoff: float=12.8) -> list[int, int, int]:
        """Method to calculate Unitcells (for periodic boundary condition)"""
        unit_cell = atoms.cell[:]
        #Unit cell vectors
        A = unit_cell[0]
        B = unit_cell[1]
        C = unit_cell[2]
        #minimum distances between unit cell faces
        Wa = np.divide(np.linalg.norm(np.dot(np.cross(B,C),A)), np.linalg.norm(np.cross(B,C)))
        Wb = np.divide(np.linalg.norm(np.dot(np.cross(C,A),B)), np.linalg.norm(np.cross(C,A)))
        Wc = np.divide(np.linalg.norm(np.dot(np.cross(A,B),C)), np.linalg.norm(np.cross(A,B)))

        uc_x = int(np.ceil(cutoff/(0.5*Wa)))
        uc_y = int(np.ceil(cutoff/(0.5*Wb)))
        uc_z = int(np.ceil(cutoff/(0.5*Wc)))

        return [uc_x, uc_y, uc_z]

    def _write_cif(self, atoms: ase.Atoms, out_dir: str, name: str):
        """Write a CIF file with partial charges that gRASPA can read from ASE atoms."""
        with open(os.path.join(out_dir, name), "w") as fp:
            fp.write(f"MOFA-{name}\n")    

            a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
            fp.write(f"_cell_length_a      {a}\n")
            fp.write(f"_cell_length_b      {b}\n")
            fp.write(f"_cell_length_c      {c}\n")
            fp.write(f"_cell_angle_alpha   {alpha}\n")
            fp.write(f"_cell_angle_beta    {beta}\n")
            fp.write(f"_cell_angle_gamma   {gamma}\n")
            fp.write(f"\n")
            fp.write("_symmetry_space_group_name_H-M    'P 1'\n")
            fp.write("_symmetry_int_tables_number        1\n")
            fp.write('\n')

            fp.write("loop_\n")
            fp.write("  _symmetry_equiv_pos_as_xyz\n")
            fp.write("  'x, y, z'\n")
            fp.write("loop_\n")
            fp.write("  _atom_site_label\n")
            fp.write("  _atom_site_occupancy\n")
            fp.write("  _atom_site_fract_x\n")
            fp.write("  _atom_site_fract_y\n")
            fp.write("  _atom_site_fract_z\n")
            fp.write("  _atom_site_thermal_displace_type\n")
            fp.write("  _atom_site_B_iso_or_equiv\n")
            fp.write("  _atom_site_type_symbol\n")
            fp.write("  _atom_site_charge\n")

            coords = atoms.get_scaled_positions().tolist()
            symbols = atoms.get_chemical_symbols()
            occupancies = [1 for i in range(len(symbols))] # No partial occupancy
            charges = atoms.info["_atom_site_charge"]

            no = {}

            for symbol, pos, occ, charge in zip(symbols, coords, occupancies, charges):
                if symbol in no:
                    no[symbol] += 1
                else:
                    no[symbol] = 1

                fp.write(f"{symbol}{no[symbol]} {occ:.1f} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} Biso 1.0 {symbol} {charge:.6f}\n")

    def _get_cif_from_chargemol(self, cp2k_path: str, chargemol_fname: str = 'DDEC6_even_tempered_net_atomic_charges.xyz'):
        """Return an ASE Atom object from Chargemol output"""
        with open(os.path.join(cp2k_path, chargemol_fname), "r") as f:
            symbols = []
            x = []
            y = []
            z = []
            charges = []
            positions = []
            lines = f.readlines()

            for i, line in enumerate(lines):
                if i == 0:
                    natoms = int(line)
                elif i == 1:
                    data = line.split()
                    a1 = float(data[10])
                    a2 = float(data[11])
                    a3 = float(data[12])
                    b1 = float(data[15])
                    b2 = float(data[16])
                    b3 = float(data[17])
                    c1 = float(data[20])
                    c2 = float(data[21])
                    c3 = float(data[22])

                elif i <= natoms + 1:
                    data = line.split()
                    symbols.append(data[0])
                    x = float(data[1])
                    y = float(data[2])
                    z = float(data[3])
                    charges.append(float(data[4]))
                    positions.append([x, y, z])
        cell = [[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]
        atoms = ase.Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)  
        atoms.info['_atom_site_charge'] = charges

        return atoms

    def _write_cif_from_chargemol(self, cp2k_path: str, out_dir: str, name: str, chargemol_fname: str = 'DDEC6_even_tempered_net_atomic_charges.xyz'):
        """Write a CIF file with partial charges for gRASPA using chargemol output"""
        atoms = self._get_cif_from_chargemol(cp2k_path, chargemol_fname=chargemol_fname)
        self._write_cif(atoms, out_dir, name)

    def run_graspa(self, name: str, cp2k_path: str, adsorbate: str, temperature: float, pressure: float, 
                   cutoff: float = 12.8, n_cycle: int = 500, write_output=True, unit='g/L', chargemol_fname: str='DDEC6_even_tempered_net_atomic_charges.xyz'):
        out_dir = self.run_dir / f'{name}_{adsorbate}_{temperature}_{pressure:0e}'
        out_dir.mkdir(parents=True, exist_ok=True)

        atoms = self._get_cif_from_chargemol(cp2k_path, chargemol_fname=chargemol_fname)
        sed_option = "-i ''" if platform.system() == "Darwin" else "-i" # For MacOS
        # Copy input files from templates
        subprocess.run(f"cp {_file_dir}/* {out_dir}/", shell=True)

        # Modify simulation.input for each run
        subprocess.run(f"sed {sed_option} 's/NCYCLE/{n_cycle}/g' {out_dir}/simulation.input", shell=True)
        subprocess.run(f"sed {sed_option} 's/ADSORBATE/{adsorbate}/' {out_dir}/simulation.input", shell=True)
        subprocess.run(f"sed {sed_option} 's/TEMPERATURE/{temperature}/' {out_dir}/simulation.input", shell=True)
        subprocess.run(f"sed {sed_option} 's/PRESSURE/{pressure}/' {out_dir}/simulation.input", shell=True)
        [uc_x, uc_y, uc_z] = self._calculate_cell_size(atoms=atoms)
        subprocess.run(f"sed {sed_option} 's/UC_X UC_Y UC_Z/{uc_x} {uc_y} {uc_z}/' {out_dir}/simulation.input", shell=True)
        subprocess.run(f"sed {sed_option} 's/CUTOFF/{cutoff}/g' {out_dir}/simulation.input", shell=True)

        # CIF file name
        self._write_cif(atoms, out_dir=out_dir, name=name + '.cif') # Not sure if name should have .cif or not
        #self._write_cif_from_chargemol(cp2k_path, out_dir, name)
        subprocess.run(f"sed {sed_option} 's/CIF/{name}/' {out_dir}/simulation.input", shell=True)
        
        # gRASPA outputs different units. Select the correct value based on requested unit.
        if unit == 'g/L':
            value_index = -3
        elif unit == 'mg/g':
            value_index = 7
        elif unit == 'mol/kg':
            value_index = -5
        else:
            raise ValueError("Current implementation only support output unit in 'g/L' or 'mg/g' or 'mol/kg' ")

        if write_output == True:
            self.graspa_command = self.graspa_command + " > raspa.err 2> raspa.log" # Store outputs in raspa.log. Some minor outputs are in raspa.err.
            subprocess.run(self.graspa_command, shell=True, cwd=out_dir)

            # Get output from raspa.log file
            data = subprocess.check_output("grep 'Overall: Average' {}/{}".format(out_dir, "raspa.log"), shell=True).decode('ascii')
            results = data.strip().split('\n')
            results_uptake = results[value_index]
            r = results_uptake.split(',')
            uptake = r[0].split()[-1]
            error = r[1].split()[-1]

            return float(uptake), float(error)
        else:
            import sys
            process = subprocess.run(self.graspa_command, shell=True, cwd=out_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
            stderr_output = process.stderr # gRASPA main outputs are in stderr
            results = [line for line in stderr_output.split('\n') if "Overall: Average" in line]

            if results:
                results_uptake = results[value_index]  # Adjust value_index accordingly based on unit
                r = results_uptake.split(',')
                uptake = r[0].split()[-1]
                error = r[1].split()[-1]
                
                return float(uptake), float(error)
            else:
                raise ValueError("No relevant output found in stderr.")


