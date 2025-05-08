"""Functions used across multiple RASPA versions"""
from pathlib import Path

import numpy as np
import ase


def calculate_cell_size(atoms: ase.Atoms, cutoff: float = 12.8) -> tuple[int, int, int]:
    """Method to calculate Unitcells (for periodic boundary condition) for GCMC

    Args:
        atoms (ase.Atoms): ASE atom object
        cutoff (float, optional): Cutoff in Angstrom. Defaults to 12.8.

    Returns:
        list[int, int, int]: Unit cell in x, y and z
    """
    unit_cell = atoms.cell[:]
    # Unit cell vectors
    a = unit_cell[0]
    b = unit_cell[1]
    c = unit_cell[2]
    # minimum distances between unit cell faces
    wa = np.divide(np.linalg.norm(np.dot(np.cross(b, c), a)), np.linalg.norm(np.cross(b, c)))
    wb = np.divide(np.linalg.norm(np.dot(np.cross(c, a), b)), np.linalg.norm(np.cross(c, a)))
    wc = np.divide(np.linalg.norm(np.dot(np.cross(a, b), c)), np.linalg.norm(np.cross(a, b)))

    uc_x = int(np.ceil(cutoff / (0.5 * wa)))
    uc_y = int(np.ceil(cutoff / (0.5 * wb)))
    uc_z = int(np.ceil(cutoff / (0.5 * wc)))

    return uc_x, uc_y, uc_z


def write_cif(atoms: ase.Atoms, out_dir: Path, name: str):
    """Save a CIF file with partial charges for RASPA2 from an ASE Atoms object.

    Args:
        atoms (ase.Atoms): ASE Atoms object containing partial charges in atoms.info["_atom_site_charge"].
        out_dir (str): Directory to save the output file.
        name (str): Name of the output file.
    """

    with open(out_dir / name, "w") as fp:
        fp.write(f"MOFA-{name}\n")

        a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
        fp.write(f"_cell_length_a      {a}\n")
        fp.write(f"_cell_length_b      {b}\n")
        fp.write(f"_cell_length_c      {c}\n")
        fp.write(f"_cell_angle_alpha   {alpha}\n")
        fp.write(f"_cell_angle_beta    {beta}\n")
        fp.write(f"_cell_angle_gamma   {gamma}\n")
        fp.write("\n")
        fp.write("_symmetry_space_group_name_H-M    'P 1'\n")
        fp.write("_symmetry_int_tables_number        1\n")
        fp.write("\n")

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
        occupancies = [
            1 for i in range(len(symbols))
        ]  # No partial occupancy
        charges = atoms.info["_atom_site_charge"]

        no = {}

        for symbol, pos, occ, charge in zip(
                symbols, coords, occupancies, charges
        ):
            if symbol in no:
                no[symbol] += 1
            else:
                no[symbol] = 1

            fp.write(
                f"{symbol}{no[symbol]} {occ:.1f} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} Biso 1.0 {symbol} {charge:.6f}\n"
            )


def get_cif_from_chargemol(
    cp2k_path: Path,
    chargemol_fname: str = "DDEC6_even_tempered_net_atomic_charges.xyz"
) -> ase.Atoms:
    """Return an ASE atom object from a Chargemol output file.

    Args:
        cp2k_path (str): Path to Chargemol output.
        chargemol_fname (str, optional): Chargemol output filename. Defaults to "DDEC6_even_tempered_net_atomic_charges.xyz".

    Returns:
        ase.Atoms: ASE Atoms object containing partial charges in atoms.info["_atom_site_charge"].
    """
    with open(cp2k_path / chargemol_fname, "r") as f:
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
    atoms = ase.Atoms(
        symbols=symbols, positions=positions, cell=cell, pbc=True
    )
    atoms.info["_atom_site_charge"] = charges

    return atoms
