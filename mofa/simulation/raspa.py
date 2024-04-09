"""Simulation operations that involve RASPA"""
from typing import Sequence
from subprocess import run
from pathlib import Path
import os
import sys

import ase
import io
import shutil
import logging
import pandas as pd
from ase.geometry.cell import cell_to_cellpar

from .cif2lammps.main_conversion import single_conversion
from .cif2lammps.UFF4MOF_construction import UFF4MOF


logger = logging.getLogger(__name__)


def read_lmp_sec_str2df(df_str, comment_char="#"):
    df_str_list = list(filter(None, df_str.split("\n")))
    df_str_list = [x.strip() for x in df_str_list]
    rows, comments = list(map(list, zip(*[[pd.read_csv(io.StringIO(x.split(comment_char)[0].strip(
    )), sep=r"\s+", header=None), x.split(comment_char)[1].strip()] for x in df_str_list])))
    df = pd.concat(rows, axis=0).fillna("")
    df[comment_char] = comment_char
    df["comment"] = comments
    return df.reset_index(drop=True)


def write_pseudo_atoms_def(
        raspa_path: str,
        ff_style_dict: dict,
        mass_df: pd.DataFrame,
        atom_df: pd.DataFrame) -> (str):
    """Use LAMMPS input files to write pseudo_atoms.def

    Args:
        raspa_path: output directory
        mass_df: LAMMPS Masses table in Pandas format
        atom_df: LAMMPS Atoms table in Pandas format
    Returns:
        raspa_file_name: written file name
    """
    mass_df.columns = ["type", "mass", "#", "comment"]
    atom_df["mass"] = atom_df["type"].map(
        dict(zip(mass_df["type"].to_list(), mass_df["mass"].to_list())))
    atom_df["element"] = atom_df["type"].map(
        dict(zip(mass_df["type"].to_list(), mass_df["comment"].to_list())))
    _pseudo_atoms_df = pd.read_csv(
        io.StringIO("""He     yes  He He 0  4.002602    0.0      0.0    1.0   1.0    0     0   relative    0
C_co2  yes  C  C  0  12.0        0.6512   0.0    1.0   0.720  0     0   relative    0
O_co2  yes  O  O  0  15.9994    -0.3256   0.0    1.0   0.68   0     0   relative    0
O_o2   yes  O  O  0  15.9994    -0.112    0.0    1.0   0.7    0     0   relative    0
O_com  no   O  -  0  0.0         0.224    0.0    1.0   0.7    0     0   relative    0
N_n2   yes  N  N  0  14.00674   -0.405    0.0    1.0   0.7    0     0   relative    0
N_com  no   N  -  0  0.0         0.810    0.0    1.0   0.7    0     0   relative    0
Ar     yes  Ar Ar 0  39.948      0.0      0.0    1.0   0.7    0     0   relative    0
CH4    yes  C  C  0  16.04246    0.0      0.0    1.0   1.00   0     0   relative    0
CH3    yes  C  C  0  15.03452    0.0      0.0    1.0   1.00   0     0   relative    0
CH2    yes  C  C  0  14.02658    0.0      0.0    1.0   1.00   0     0   relative    0
CH     yes  C  C  0  13.01864    0.0      0.0    1.0   1.00   0     0   relative    0
C      yes  C  C  0  12.0        0.0      0.0    1.0   1.00   0     0   relative    0
"""),
        sep=r"\s+",
        header=None,
        names=[
            "#type",
            "print",
            "as",
            "chem",
            "oxidation",
            "mass",
            "charge",
            "polarization",
            "B-factor",
            "radii",
            "connectivity",
            "anisotropic",
            "anisotropic-type",
            "tinker-type"])
    pseudo_atoms_df_header_str = "#type      print   as    chem  oxidation   mass" + "        " + \
                                 "charge   polarization B-factor radii  connectivity anisotropic anisotropic-type   tinker-type"
    pseudo_atoms_df = atom_df[["comment", "element", "element", "mass", "q"]].copy(
        deep=True).reset_index(drop=True)
    pseudo_atoms_df.columns = ["#type", "as", "chem", "mass", "charge"]
    pseudo_atoms_df["print"] = "yes"
    pseudo_atoms_df["oxidation"] = "0"
    pseudo_atoms_df["polarization"] = "0.0"
    pseudo_atoms_df["B-factor"] = "1.0"
    pseudo_atoms_df["radii"] = "1.00"
    pseudo_atoms_df["connectivity"] = "0"
    pseudo_atoms_df["anisotropic"] = "0"
    pseudo_atoms_df["anisotropic-type"] = "relative"
    pseudo_atoms_df["tinker-type"] = "0"
    pseudo_atoms_df = pseudo_atoms_df[_pseudo_atoms_df.columns]
    pseudo_atoms_df = pd.concat(
        [pseudo_atoms_df, _pseudo_atoms_df], axis=0).reset_index(drop=True)
    pseudo_atoms_str = "#number of pseudo atoms\n" + "%d" % len(pseudo_atoms_df) + "\n" + pseudo_atoms_df_header_str + "\n" + \
                       "\n".join(["{:<10} {:<7} {:<5} {:<5} {:<11} {:<10} {:<9} {:<12} {:<8} {:<6} {:<12} {:<11} {:<18} {:<11}".format(
                           pseudo_atoms_df.at[x, "#type"].strip(),
                           pseudo_atoms_df.at[x, "print"],
                           pseudo_atoms_df.at[x, "as"],
                           pseudo_atoms_df.at[x, "chem"],
                           pseudo_atoms_df.at[x, "oxidation"],
                           pseudo_atoms_df.at[x, "mass"],
                           "%1.4f" % pseudo_atoms_df.at[x, "charge"],
                           pseudo_atoms_df.at[x, "polarization"],
                           pseudo_atoms_df.at[x, "B-factor"],
                           pseudo_atoms_df.at[x, "radii"],
                           pseudo_atoms_df.at[x, "connectivity"],
                           pseudo_atoms_df.at[x, "anisotropic"],
                           pseudo_atoms_df.at[x, "anisotropic-type"],
                           pseudo_atoms_df.at[x, "tinker-type"]) for x in pseudo_atoms_df.index])

    with open(os.path.join(raspa_path, "pseudo_atoms.def"), "w") as wf:
        wf.write(pseudo_atoms_str)


def write_force_field_mixing_rules_def(
        raspa_path: str,
        ff_style_dict: dict,
        pair_coeff_df: pd.DataFrame,
        atom_df: pd.DataFrame) -> (str):
    """Use LAMMPS input files to write pseudo_atoms.def

    Args:
        raspa_path: output directory
        pair_coeff_df: LAMMPS Pair_Coeff table in Pandas format
        atom_df: LAMMPS Atoms table in Pandas format
    Returns:
        raspa_file_name: written file name
    """
    NAvogadro = 6.02214076e23
    kB = 1.380649e-23
    kCal2Joule = 4184
    lammps2raspa_energy = kCal2Joule / (NAvogadro * kB)
    _mixing_df = read_lmp_sec_str2df(
        """He             lennard-jones    10.9      2.64""" +
        "         " + """// J.O. Hirschfelder et al., Molecular Theory of Gases and Liquids, Wiley, New York, 1954, p. 1114.
O_co2          lennard-jones    85.671    3.017        // A. Garcia-Sanchez et al., J. Phys. Chem. C 2009, 113, 8814-8820.
C_co2          lennard-jones    29.933    2.745        // idem
N_n2           lennard-jones    38.298    3.306        // A. Martin-Calvo et al. , Phys. Chem. Chem. Phys. 2011, 13, 11165-11174.
N_com          none                                    // idem
O_o2           lennard-jones    53.023    3.045        // A. Martin-Calvo et al. , Phys. Chem. Chem. Phys. 2011, 13, 11165-11174.
O_com          none                                    //
Ar             lennard-jones   124.070    3.38         // A. Martin-Calvo et al. , Phys. Chem. Chem. Phys. 2011, 13, 11165-11174.
CH4            lennard-jones    158.5     3.72         // M. G. Martin et al., J. Chem. Phys. 2001, 114, 7174-7181.
CH3            lennard-jones    108.0     3.76         // D. Dubbeldam et al., J. Phys. Chem. B, 108(33), 12301-12313
CH2            lennard-jones    56.0      3.96         // idem
CH             lennard-jones    17.0      4.67         // idem
C              lennard-jones     0.8      6.38         // idem
""", comment_char="//")
    _mixing_df.columns = [
        "atom",
        "type",
        "eps(K)",
        "sig(Ang)",
        "//",
        "comment"]

    pair_coeff_df.columns = [
        "type",
        "eps(kCal/mol)",
        "sig(Ang)",
        "#",
        "comment"]
    pair_coeff_df["eps(K)"] = [
        "%.4f" %
        x for x in (
            pair_coeff_df["eps(kCal/mol)"] *
            lammps2raspa_energy)]
    pair_coeff_df["sig(Ang)"] = ["%.4f" %
                                 x for x in pair_coeff_df["sig(Ang)"]]
    atom_df["eps(K)"] = atom_df["type"].map(
        dict(zip(pair_coeff_df["type"], pair_coeff_df["eps(K)"])))
    atom_df["sig(Ang)"] = atom_df["type"].map(
        dict(zip(pair_coeff_df["type"], pair_coeff_df["sig(Ang)"])))
    mixing_df = atom_df[["comment", "eps(K)", "sig(Ang)"]].copy(
        deep=True).reset_index(drop=True)
    mixing_df.columns = ["atom", "eps(K)", "sig(Ang)"]
    mixing_df["type"] = "lennard-jones"
    mixing_df["//"] = "//"
    mixing_df["comment"] = "UFF4MOF"
    mixing_df = pd.concat([mixing_df, _mixing_df],
                          axis=0).reset_index(drop=True)
    mixing_str = "\n".join(["{:<14} {:<16} {:<9} {:<12} {:<3} {:<100}".format(
        mixing_df.at[x, "atom"].strip(),
        mixing_df.at[x, "type"].strip(),
        mixing_df.at[x, "eps(K)"],
        mixing_df.at[x, "sig(Ang)"],
        mixing_df.at[x, "//"].strip(),
        mixing_df.at[x, "comment"].strip(),
    ) for x in mixing_df.index])
    mixing_method = "Lorentz-Berthelot"
    if "geometric" in ff_style_dict["pair_modify"]:
        mixing_method = "Jorgensen"
    force_field_mixing_rules_str = """# general rule for shifted vs truncated
shifted
# general rule tailcorrections
yes
# number of defined interactions
""" + "%d" % len(mixing_df) + """
# type interaction
""" + mixing_str + """
# general mixing rule for Lennard-Jones
""" + mixing_method + """
"""
    with open(os.path.join(raspa_path, "force_field_mixing_rules.def"), "w") as wf:
        wf.write(force_field_mixing_rules_str)


class RASPARunner:
    """Interface for running pre-defined RASPA workflows

    Args:
        raspa_command: Command used to launch RASPA
        raspa_sims_root_path: Scratch directory for RASPA simulations
        delete_finished: Whether to delete run files once completed
    """

    def __init__(self,
                 raspa_command: Sequence[str] = (Path(sys.prefix) / "bin" / "simulate",),
                 raspa_sims_root_path: str = "raspa_sims",
                 delete_finished: bool = True):
        self.raspa_command = raspa_command
        self.raspa_sims_root_path = raspa_sims_root_path
        os.makedirs(self.raspa_sims_root_path, exist_ok=True)
        self.delete_finished = delete_finished

    def prep_common_files(self, run_name: str, raspa_path: str | Path, mof_ase_atoms: ase.Atoms):
        # MOF cif file with partial charge and labeled in RASPA convention
        cifdf = pd.DataFrame(mof_ase_atoms.get_scaled_positions(), columns=["xs", "ys", "zs"])
        cifdf["q"] = mof_ase_atoms.arrays["q"]
        cifdf["el"] = mof_ase_atoms.symbols
        a, b, c, alpha, beta, gamma = cell_to_cellpar(mof_ase_atoms.cell)

        label_map = {}
        for val, subdf in cifdf.groupby('el'):
            newmap = dict(zip(subdf.index.tolist(), subdf["el"] + [str(x) for x in range(0, len(subdf))]))
            if isinstance(label_map, type(None)):
                label_map = dict(newmap)
            else:
                label_map.update(newmap)
        cifdf["label"] = cifdf.index.map(label_map)
        cifdf = cifdf[["label", "el", "xs", "ys", "zs", "q"]]
        cifstr = """
_audit_author_name 'xyan11@uic.edu'

_cell_length_a       """ + "%.8f" % a + """
_cell_length_b       """ + "%.8f" % b + """
_cell_length_c       """ + "%.8f" % c + """
_cell_angle_alpha    """ + "%.8f" % alpha + """
_cell_angle_beta     """ + "%.8f" % beta + """
_cell_angle_gamma    """ + "%.8f" % gamma + """
_cell_volume         """ + "%.8f" % mof_ase_atoms.cell.volume + """

_symmetry_cell_setting             triclinic
_symmetry_space_group_name_Hall    'P 1'
_symmetry_space_group_name_H-M     'P 1'
_symmetry_Int_Tables_number        1

loop_
_symmetry_equiv_pos_as_xyz
 'x,y,z'

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_charge
""" + cifdf.to_string(header=None, index=None) + "\n"
        # Write the cif file to disk
        with open(Path(raspa_path) / f'{run_name}.cif', "w") as wf:
            wf.write(cifstr)

        # meta information about force field
        with open(Path(raspa_path) / "force_field.def", "w") as wf:
            wf.write("""# rules to overwrite
0
# number of defined interactions
0
# mixing rules to overwrite
0
""")

        cif_path = os.path.join(raspa_path, f'{run_name}.cif')
        cif_path_ase = os.path.join(raspa_path, f'{run_name}-ase.cif')
        mof_ase_atoms.write(cif_path_ase, 'cif')
        try:
            single_conversion(cif_path_ase,
                              force_field=UFF4MOF,
                              ff_string='UFF4MOF',
                              small_molecule_force_field=None,
                              outdir=raspa_path,
                              charges=False,
                              parallel=False,
                              replication='1x1x1',
                              read_cifs_pymatgen=True,
                              add_molecule=None,
                              small_molecule_file=None)
            in_file_name = [x for x in os.listdir(raspa_path) if x.startswith("in.") and not x.startswith("in.lmp")][0]
            data_file_name = [x for x in os.listdir(raspa_path) if x.startswith("data.") and not x.startswith("data.lmp")][0]
            logger.info("Reading data file for element list: " + os.path.join(raspa_path, data_file_name))

        except Exception as e:
            shutil.rmtree(raspa_path)
            raise e

        # parse output
        read_str = None
        with io.open(os.path.join(raspa_path, data_file_name), "r") as rf2:
            read_str = rf2.read()
        mass_df = read_lmp_sec_str2df(read_str.split(
            "Masses")[1].split("Pair Coeffs")[0].strip())
        pair_coeff_df = read_lmp_sec_str2df(read_str.split(
            "Pair Coeffs")[1].split("Bond Coeffs")[0].strip())
        atom_df = read_lmp_sec_str2df(read_str.split("$$$atoms$$$")[
            1].split("Bonds")[0].strip())
        atom_df.columns = [
            'id', 'mol', 'type', 'q', 'x', 'y', 'z', '#', "comment"]
        _atom_df = pd.read_csv(
            io.StringIO(
                "\n".join(
                    atom_df["comment"].to_list())), sep=r"\s+", header=None, index_col=None, names=[
                "comment", "fx", "fy", "fz"])
        atom_df = pd.concat([atom_df[['id', 'mol', 'type', 'q', 'x', 'y', 'z', '#']].reset_index(
            drop=True), _atom_df.reset_index(drop=True)], axis=1)

        ff_style_dict = None
        with io.open(os.path.join(raspa_path, in_file_name), "r") as rf1:
            ff_style_str = rf1.read()
            ff_style_list = list(filter(None, [x.strip() for x in ff_style_str.split("\n")]))
            ff_style_list = [
                pd.read_csv(
                    io.StringIO(x),
                    header=None,
                    sep=r"\s+").values.tolist()[0] for x in ff_style_list]
            ff_style_list = [[x[0], x[1:]] for x in ff_style_list]
            ff_style_dict = dict(zip(*list(map(list, zip(*ff_style_list)))))

        write_pseudo_atoms_def(
            raspa_path, ff_style_dict, mass_df, atom_df)
        write_force_field_mixing_rules_def(
            raspa_path, ff_style_dict, pair_coeff_df, atom_df)
        os.remove(os.path.join(raspa_path, in_file_name))
        os.remove(os.path.join(raspa_path, data_file_name))
        os.remove(cif_path_ase)

    def run_GCMC_single(self, mof_ase_atoms: ase.Atoms, run_name: str, temperature_K: float = 300., pressure_Pa: float = 1e4,
                        stepsize_fs: float = 0.5, timesteps: int = 200000, report_frequency: int = 1000,
                        cell_rep: list[int] = [2, 2, 2]) -> list[float]:
        """Use cif2lammps to assign force field to a single MOF and generate input files for raspa simulation

        Args:
            mof_ase_atoms: ase.Atoms object with charge information
            run_name: Name of the run directory
            temperature_K: Temperature
            pressure_Pa: Pressure
            timesteps: Number of timesteps to run
            report_frequency: How often to report structures
            cell_rep: replicate unit cell
        Returns:
            raspa_path: a directory with the raspa simulation input files
        """

        # Convert the cif_path to string, as that's what the underlying library uses
        raspa_path = os.path.join(self.raspa_sims_root_path, run_name)
        os.makedirs(raspa_path, exist_ok=True)
        self.prep_common_files(run_name, raspa_path, mof_ase_atoms)
        with open(Path(raspa_path) / "helium.def", "w") as wf:
            wf.write("""# critical constants: Temperature [T], Pressure [Pa], and Acentric factor [-]
5.2
228000.0
-0.39
# Number Of Atoms
1
# Number Of Groups
1
# Alkane-group
flexible
# number of atoms
1
# atomic positions
0 He
# Chiral centers Bond  BondDipoles Bend  UrayBradley InvBend  Torsion Imp. Torsion Bond/Bond """ +
                     """Stretch/Bend Bend/Bend Stretch/Torsion Bend/Torsion IntraVDW IntraCoulomb
               0    0            0    0            0       0        0            0         0 """ +
                     """           0         0               0            0        0            0
# Number of config moves
0
""")

        # He void fraction input
        with open(Path(raspa_path) / "simulation.input", "w") as wf:
            wf.write("""SimulationType                       MonteCarlo
NumberOfCycles                       """ + str(int(timesteps / 5)) + """
PrintEvery                           """ + str(report_frequency) + """
PrintPropertiesEvery                 """ + str(report_frequency) + """

Forcefield                           local

Framework 0
FrameworkName """ + f'{run_name}' + """
UnitCells """ + "%d" % cell_rep[0] + " " + "%d" % cell_rep[1] + " " + "%d" % cell_rep[2] + """
ExternalTemperature """ + "%.2f" % temperature_K + """

Component 0 MoleculeName             helium
            MoleculeDefinition       local
            WidomProbability         1.0
            CreateNumberOfMolecules  0
""")
        # run He void calcultion
        with open(Path(raspa_path) / 'stdout_he_void.raspa', 'w') as fp, open(Path(raspa_path) / 'stderr_he_void.raspa', 'w') as fe:
            env = None
            run(list(self.raspa_command), cwd=raspa_path, stdout=fp, stderr=fe, env=env)

        # parse output
        outdir = os.path.join(raspa_path, "Output")
        outdir = os.path.join(outdir, os.listdir(outdir)[0])
        outfile = os.path.join(outdir, [x for x in os.listdir(outdir) if "output_" in x and x.endswith(".data")][0])
        outstr = None
        with open(outfile, "r") as rf:
            outstr = rf.read()
        He_Void_Faction = float(outstr.split("[helium] Average Widom Rosenbluth-weight:")[1].split("+/-")[0])

        os.rename(Path(raspa_path) / "simulation.input", Path(raspa_path) / "simulation-He-void.input")
        os.rename(Path(raspa_path) / "Movies", Path(raspa_path) / "Movies-He-void")
        os.rename(Path(raspa_path) / "Output", Path(raspa_path) / "Output-He-void")
        os.rename(Path(raspa_path) / "Restart", Path(raspa_path) / "Restart-He-void")
        os.rename(Path(raspa_path) / "VTK", Path(raspa_path) / "VTK-He-void")

        with open(Path(raspa_path) / "simulation.input", "w") as wf:
            wf.write("""SimulationType                MonteCarlo
NumberOfCycles                """ + "%d" % timesteps + """
NumberOfInitializationCycles  """ + "%d" % (timesteps / 10) + """
PrintEvery                    """ + "%d" % report_frequency + """
PrintPropertiesEvery          """ + "%d" % report_frequency + """
RestartFile                   no

ChargeMethod                  Ewald
CutOff                        12.0
Forcefield                    local
UseChargesFromCIFFile         yes
EwaldPrecision                1e-6
TimeStep                      """ + str(stepsize_fs / 1000) + """

Framework 0
FrameworkName """ + f'{run_name}' + """
UnitCells """ + "%d" % cell_rep[0] + " " + "%d" % cell_rep[1] + " " + "%d" % cell_rep[2] + """
HeliumVoidFraction """ + str(He_Void_Faction) + """
ExternalTemperature """ + "%.2f" % temperature_K + """
ExternalPressure """ + "%.2f" % pressure_Pa + """

Component 0 MoleculeName              CO2
    MoleculeDefinition        local
    IdealGasRosenbluthWeight  1.0
    TranslationProbability    1.0
    RotationProbability       1.0
    ReinsertionProbability    1.0
    SwapProbability           1.0
    CreateNumberOfMolecules   0

""")

        with open(Path(raspa_path) / "CO2.def", "w") as wf:
            wf.write("""# critical constants: Temperature [T], Pressure [Pa], and Acentric factor [-]
304.1282
7377300.0
0.22394
#Number Of Atoms
3
# Number of groups
1
# CO2-group
rigid
# number of atoms
3
# atomic positions
0 O_co2     0.0           0.0           1.149
1 C_co2     0.0           0.0           0.0
2 O_co2     0.0           0.0          -1.149
# Chiral centers Bond  BondDipoles Bend  UrayBradley InvBend  Torsion Imp. Torsion Bond/Bond""" + " " +
                     """Stretch/Bend Bend/Bend Stretch/Torsion Bend/Torsion IntraVDW IntraCoulomb
             0    2            0    0            0       0        0            0         0""" + "            " +
                     """0         0               0            0        0            0
# Bond stretch: atom n1-n2, type, parameters
0 1 RIGID_BOND
1 2 RIGID_BOND
# Number of config moves
0
""")
        # run CO2 GCMC
        with open(Path(raspa_path) / 'stdout_CO2_gcmc.raspa', 'w') as fp, open(Path(raspa_path) / 'stderr_CO2_gcmc.raspa', 'w') as fe:
            env = None
            run(list(self.raspa_command), cwd=raspa_path, stdout=fp, stderr=fe, env=env)
        os.rename(Path(raspa_path) / "simulation.input", Path(raspa_path) / "simulation-CO2-gcmc.input")

        # parse output
        outdir = os.path.join(raspa_path, "Output")
        outdir = os.path.join(outdir, os.listdir(outdir)[0])
        outfile = os.path.join(outdir, [x for x in os.listdir(outdir) if "output_" in x and x.endswith(".data")][0])
        outstr = None
        with open(outfile, "r") as rf:
            outstr = rf.read()
        gas_ads_info = outstr.split("Average loading excess [mol/kg framework]")[1].strip().split("[-]")[0].strip()
        gas_ads_mean, gas_ads_std = [float(x) for x in gas_ads_info.split("+/-")]
        return [gas_ads_mean, gas_ads_std]
