import os
import io
import re
import shutil
import pandas as pd
import numpy as np
from cif2lammps.main_conversion import single_conversion
from cif2lammps.UFF4MOF_construction import UFF4MOF


def read_lmp_sec_str2df(df_str, comment_char="#"):
    df_str_list = list(filter(None, df_str.split("\n")))
    df_str_list = [x.strip() for x in df_str_list]
    rows, comments = list(map(list, zip(*[[pd.read_csv(io.StringIO(x.split(comment_char)[0].strip(
    )), sep=r"\s+", header=None), x.split(comment_char)[1].strip()] for x in df_str_list])))
    df = pd.concat(rows, axis=0).fillna("")
    df[comment_char] = comment_char
    df["comment"] = comments
    return df.reset_index(drop=True)


class RASPARunner:
    """Interface for running pre-defined LAMMPS workflows

    Args:
        raspa_command: Command used to launch RASPA
    """
    raspa_command = "gcmc"
    raspa_sims_root_path = "raspa_sims"
    cif_files_root_path = "cif_files"
    cif_files_paths = []

    def __init__(
            self,
            raspa_command: str = "gcmc",
            raspa_sims_root_path: str = "raspa_sims",
            cif_files_root_path: str = "cif_files"):
        """Read cif files from input directory, make directory for raspa simulation input files

        Args:
            raspa_command: raspa simulation type, default: "gcmc"
            raspa_sims_root_path: output directory, default: "raspa_sims"
            cif_files_root_path: input directory to look for cif files: "cif_files"
        Returns:
            None
        """
        self.lmp_command = raspa_command
        self.raspa_sims_root_path = raspa_sims_root_path
        print("Making RASPA simulation root path at: " +
              os.path.join(os.getcwd(), self.raspa_sims_root_path))
        os.makedirs(self.raspa_sims_root_path, exist_ok=True)
        print(
            "Scanning cif files at: " +
            os.path.join(
                os.getcwd(),
                self.cif_files_root_path))
        self.cif_files_root_path = cif_files_root_path
        self.cif_files_paths = [
            os.path.join(
                self.cif_files_root_path,
                x) for x in os.listdir(
                self.cif_files_root_path) if x.endswith(".cif")]
        print("Found " +
              "%d" %
              len(self.cif_files_paths) +
              " files with .cif extension! \n")

    def write_pseudo_atoms_def(
            self,
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

        # pseudo_atoms_str = "#number of pseudo atoms\n" + "%d" % len(pseudo_atoms_df) + "\n" + \
        # pseudo_atoms_df.to_string(header=True, index=None, justify="left",
        # col_space=[10, 7, 5, 5, 11, 10, 9, 12, 8, 6, 12, 11, 18, 10])
        with io.open(os.path.join(raspa_path, "pseudo_atoms.def"), "w", newline="\n") as wf:
            wf.write(pseudo_atoms_str)
        return "pseudo_atoms.def"

    def write_force_field_mixing_rules_def(
            self,
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
        with io.open(os.path.join(raspa_path, "force_field_mixing_rules.def"), "w", newline="\n") as wf:
            wf.write(force_field_mixing_rules_str)
        return "force_field_mixing_rules.def"

    def write_framework_def(self, raspa_path: str,
                            bond_coeff_df: pd.DataFrame,
                            angle_coeff_df: pd.DataFrame,
                            dihedral_coeff_df: pd.DataFrame,
                            improper_coeff_df: pd.DataFrame,
                            bond_df: pd.DataFrame,
                            angle_df: pd.DataFrame,
                            dihedral_df: pd.DataFrame,
                            improper_df: pd.DataFrame) -> (str):
        """Use LAMMPS input files to write framework.def

        Args:
            raspa_path: output directory
            bond_coeff_df: LAMMPS Bond_Coeff table in Pandas format,
            angle_coeff_df: LAMMPS Angle_Coeff table in Pandas format,
            dihedral_coeff_df: LAMMPS Dihedral_Coeff table in Pandas format,
            improper_coeff_df: LAMMPS Improper_Coeff table in Pandas format,
            bond_df: LAMMPS Bonds table in Pandas format,
            angle_df: LAMMPS Angles table in Pandas format,
            dihedral_df: LAMMPS Dihedrals table in Pandas format,
            improper_df: LAMMPS Impropers table in Pandas format,
        Returns:
            raspa_file_name: written file name
        """
        NAvogadro = 6.02214076e23
        kB = 1.380649e-23
        kCal2Joule = 4184
        lammps2raspa_energy = kCal2Joule / (NAvogadro * kB)

        bond_df = bond_df.reset_index(drop=True)
        bond_coeff_df.columns = [
            "type",
            "k(kCal/mol/ang^2)",
            "r0(ang)",
            "#",
            "comment"]
        bond_coeff_df["k(K/ang^2)"] = bond_coeff_df["k(kCal/mol/ang^2)"] * \
            lammps2raspa_energy
        bond_df.columns = ["id", "type", "at1", "at2", "#", "comment"]
        bond_df["k(K/ang^2)"] = bond_df["type"].map(
            dict(zip(bond_coeff_df["type"], bond_coeff_df["k(K/ang^2)"])))
        bond_df["r0(ang)"] = bond_df["type"].map(
            dict(zip(bond_coeff_df["type"], bond_coeff_df["r0(ang)"])))
        bond_df = bond_df[["comment", "k(K/ang^2)", "r0(ang)"]].copy(deep=True)
        bond_df.columns = ["atoms", "k(K/ang^2)", "r0(ang)"]
        bond_df["type"] = "HARMONIC_BOND"
        bond_df = bond_df[["atoms",
                           "type",
                           "k(K/ang^2)",
                           "r0(ang)"]].copy(deep=True).reset_index(drop=True)
        bond_str = bond_df.to_string(header=None, index=None)

        angle_coeff_df.columns = [
            "type_id",
            "type",
            "param1",
            "param2",
            "param3",
            "param4",
            "#",
            "comment"]
        angle_coeff_fourier_df = angle_coeff_df[angle_coeff_df["type"] == "fourier"].copy(
            deep=True).reset_index(drop=True)
        angle_coeff_fourier_df.columns = [
            "type_id", "type", "K", "C0", "C1", "C2", "#", "comment"]
        angle_coeff_cos_df = angle_coeff_df[angle_coeff_df["type"] ==
                                            "cosine/periodic"].copy(deep=True).reset_index(drop=True)
        angle_coeff_cos_df.columns = [
            "type_id", "type", "C", "B", "n", "", "#", "comment"]
        angle_coeff_cos_df["p0"] = 2. * angle_coeff_cos_df["C"] / \
            angle_coeff_cos_df["n"] / angle_coeff_cos_df["n"] * lammps2raspa_energy
        angle_coeff_cos_df["p1"] = angle_coeff_cos_df["n"]
        angle_coeff_cos_df["p2"] = 0. - (90. * angle_coeff_cos_df["B"]
                                         * ((-1) ** (angle_coeff_cos_df["n"] + 1))) + 90.
        angle_coeff_cos_df["p3"] = np.nan
        angle_coeff_cos_df["type"] = "COSINE_BEND"
        angle_coeff_fourier_df["p0"] = angle_coeff_fourier_df["K"] * \
            lammps2raspa_energy
        angle_coeff_fourier_df["p1"] = angle_coeff_fourier_df["C0"]
        angle_coeff_fourier_df["p2"] = angle_coeff_fourier_df["C1"]
        angle_coeff_fourier_df["p3"] = angle_coeff_fourier_df["C2"]
        angle_coeff_fourier_df["type"] = "FOURIER_SERIES_BEND"
        angle_coeff_df = pd.concat([angle_coeff_cos_df[["type_id", "type", "p0", "p1", "p2", "p3"]], angle_coeff_fourier_df[[
                                   "type_id", "type", "p0", "p1", "p2", "p3"]]], axis=0).sort_values(by="type_id")
        angle_df = angle_df.reset_index(drop=True)
        angle_df.columns = ["id", "type", "at1", "at2", "at3", "#", "comment"]
        angle_df["p0"] = angle_df["type"].map(
            dict(zip(angle_coeff_df["type_id"], angle_coeff_df["p0"])))
        angle_df["p1"] = angle_df["type"].map(
            dict(zip(angle_coeff_df["type_id"], angle_coeff_df["p1"])))
        angle_df["p2"] = angle_df["type"].map(
            dict(zip(angle_coeff_df["type_id"], angle_coeff_df["p2"])))
        angle_df["p3"] = angle_df["type"].map(
            dict(zip(angle_coeff_df["type_id"], angle_coeff_df["p3"])))
        angle_df["type"] = angle_df["type"].map(
            dict(zip(angle_coeff_df["type_id"], angle_coeff_df["type"])))
        angle_df = angle_df[["comment", "type", "p0", "p1", "p2", "p3"]].copy(
            deep=True).reset_index(drop=True)
        angle_str = angle_df.to_string(header=None, index=None, na_rep="")

        dihedral_df = dihedral_df.reset_index(drop=True)
        dihedral_coeff_df.columns = ["type_id", "K", "d", "n", "#", "comment"]
        dihedral_coeff_df["p0"] = dihedral_coeff_df["K"] * lammps2raspa_energy
        dihedral_coeff_df["p1"] = dihedral_coeff_df["n"]
        dihedral_coeff_df["p2"] = 90. - (dihedral_coeff_df["d"] * 90.)
        dihedral_df.columns = [
            "id",
            "type",
            "at1",
            "at2",
            "at3",
            "at4",
            "#",
            "comment"]
        dihedral_df["p0"] = dihedral_df["type"].map(
            dict(zip(dihedral_coeff_df["type_id"], dihedral_coeff_df["p0"])))
        dihedral_df["p1"] = dihedral_df["type"].map(
            dict(zip(dihedral_coeff_df["type_id"], dihedral_coeff_df["p1"])))
        dihedral_df["p2"] = dihedral_df["type"].map(
            dict(zip(dihedral_coeff_df["type_id"], dihedral_coeff_df["p2"])))
        dihedral_df["type"] = "CVFF_DIHEDRAL"
        dihedral_df = dihedral_df[["comment", "type", "p0", "p1", "p2"]].copy(
            deep=True).reset_index(drop=True)
        dihedral_str = dihedral_df.to_string(
            header=None, index=None, na_rep="")

        improper_df = improper_df.reset_index(drop=True)
        improper_coeff_df.columns = [
            "type_id", "K", "C0", "C1", "C2", "all", "#", "comment"]
        imp_df = pd.read_csv(
            io.StringIO(
                "\n".join(
                    improper_df["comment"].to_list())),
            sep=r"\s+",
            header=None,
            names=[
                "atI",
                "atJ",
                "atK",
                "atL"])
        imp_df = imp_df[["atJ", "atI", "atK", "atL"]]
        improper_df["comment"] = imp_df["atJ"] + "   " + \
            imp_df["atI"] + "   " + imp_df["atK"] + "   " + imp_df["atL"]
        improper_df.columns = [
            "id",
            "type_id",
            "atI",
            "atJ",
            "atK",
            "atL",
            "#",
            "comment"]
        improper_df["type"] = "TRAPPE_IMPROPER_DIHEDRAL"
        improper_df["K"] = improper_df["type_id"].map(dict(
            zip(improper_coeff_df["type_id"], improper_coeff_df["K"]))) * lammps2raspa_energy
        improper_df["C0"] = improper_df["type_id"].map(
            dict(zip(improper_coeff_df["type_id"], improper_coeff_df["C0"])))
        improper_df["C1"] = improper_df["type_id"].map(
            dict(zip(improper_coeff_df["type_id"], improper_coeff_df["C1"])))
        improper_df["C2"] = improper_df["type_id"].map(
            dict(zip(improper_coeff_df["type_id"], improper_coeff_df["C2"])))
        improper_df["p0"] = improper_df["K"] * \
            (improper_df["C0"] - improper_df["C1"] + improper_df["C2"])
        improper_df["p1"] = improper_df["K"] * improper_df["C1"]
        improper_df["p2"] = 0. - (improper_df["K"] * improper_df["C2"])
        improper_df["p3"] = 0.
        improper_df = improper_df[['comment', 'type', 'p0', 'p1', 'p2', 'p3']].copy(
            deep=True).reset_index(drop=True)
        improper_str = improper_df.to_string(
            header=None, index=None, na_rep="")

        header_df = pd.read_csv(
            io.StringIO(
                """#CoreShells bond  BondDipoles UreyBradley bend  inv  tors""" +
                " " +
                """improper-torsion bond/bond bond/bend bend/bend stretch/torsion bend/torsion
0  """ +
                "%d" %
                len(bond_df) +
                """  0  0  """ +
                "%d" %
                len(angle_df) +
                """  0  """ +
                "%d" %
                len(dihedral_df) +
                """  """ +
                "%d" %
                len(improper_df) +
                """  0  0  0  0  0"""),
            sep=r"\s+")

        framework_str = header_df.to_string(header=True, index=None) + """
#bond stretch atom n1-n2, equilibrium distance, bondforce-constant
""" + bond_str + """
#bond bending atom n1-n2-n3, equilibrium angle, bondforce-constant
""" + angle_str + """
#torsion atom n1-n2-n3-n4,
""" + dihedral_str + """
# improper torsion atom n1-n2-n3-n4,
""" + improper_str

        with io.open(os.path.join(raspa_path, "framework.def"), "w", newline="\n") as wf:
            wf.write(framework_str)
        return "framework.def"

    def rewrite_cif(
            self,
            raspa_path: str,
            cif_name: str,
            atom_df: pd.DataFrame,
            cifbox: str) -> str:
        """Use LAMMPS input files to write framework.def

        Args:
            raspa_path: output directory
            cif_name: original cif file name
            atom_df: LAMMPS Atoms table in Pandas format,
            cifbox: encoded string of cif box information,
        Returns:
            raspa_file_name: written file name
        """
        atom_df["element"] = [re.sub("[0-9]+", "", x)
                              for x in atom_df["comment"]]
        atom_str = atom_df[["comment", "element", "fx", "fy", "fz", "q"]].to_string(
            header=None, index=None, col_space=[10, 8, 20, 20, 20, 10], justify="left")
        a, b, c, alpha, beta, gamma = [
            float(x) for x in cifbox.split(":")[1].strip().split(",")]
        cosa = np.cos(np.deg2rad(alpha))
        cosb = np.cos(np.deg2rad(beta))
        cosr = np.cos(np.deg2rad(gamma))
        V = a * b * c * np.sqrt(1. - (2. * cosa * cosb * cosr) -
                                (cosa * cosa) - (cosb * cosb) - (cosr * cosr))
        new_cif_str = cif_name + """

_audit_author_name 'xyan11@uic.edu'

_cell_length_a       """ + "%.8f" % a + """
_cell_length_b       """ + "%.8f" % b + """
_cell_length_c       """ + "%.8f" % c + """
_cell_angle_alpha    """ + "%.8f" % alpha + """
_cell_angle_beta     """ + "%.8f" % beta + """
_cell_angle_gamma    """ + "%.8f" % gamma + """
_cell_volume         """ + "%.8f" % V + """

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
""" + atom_str
        new_cif_name = "mof.cif"
        with io.open(os.path.join(raspa_path, new_cif_name), "w", newline="\n") as wf:
            wf.write(new_cif_str)
        return new_cif_name

    def write_force_field_def(self, raspa_path: str) -> (str):
        """Use LAMMPS input files to write force_field.def

        Args:
            raspa_path: output directory
        Returns:
            raspa_file_name: written file name
        """
        with io.open(os.path.join(raspa_path, "force_field.def"), "w", newline="\n") as wf:
            wf.write("""# rules to overwrite
0
# number of defined interactions
0
# mixing rules to overwrite
0
""")
        return "force_field.def"

    def prep_raspa_single(
            self,
            cif_path: str,
            timesteps: int,
            report_frequency: int,
            stepsize_fs: float = 0.5,
            cell_replicate: list = [
                2,
                2,
                2],
            raspa_abs_path: str = "/projects/bbke/xyan11/conda-envs/gcmc2-310/bin/simulate") -> (
                str,
            int):
        """Use cif2lammps to assign force field to a single MOF and generate input files for raspa simulation

        Args:
            cif_path: starting structure's cif file path
            timesteps: Number of timesteps to run
            report_frequency: How often to report structures
        Returns:
            raspa_path: a directory with the raspa simulation input files
            return_code: cif2lammps running status, 0 means success (directory raspa_path will be kept),
                         -1 means failure (directory raspa_path will be destroyed)
        """
        cif_name = os.path.split(cif_path)[-1]
        raspa_path = os.path.join(
            self.raspa_sims_root_path,
            cif_name.replace(
                ".cif",
                ""))
        os.makedirs(raspa_path, exist_ok=True)
        try:
            single_conversion(cif_path,
                              force_field=UFF4MOF,
                              ff_string='UFF4MOF',
                              small_molecule_force_field=None,
                              outdir=raspa_path,
                              charges=True,
                              parallel=False,
                              replication='1x1x1',
                              read_cifs_pymatgen=True,
                              add_molecule=None,
                              small_molecule_file=None)
            in_file_name = [x for x in os.listdir(raspa_path) if x.startswith(
                "in.") and not x.startswith("in.lmp")][0]
            data_file_name = [x for x in os.listdir(raspa_path) if x.startswith(
                "data.") and not x.startswith("data.lmp")][0]
            in_file_rename = "in.lmp"
            data_file_rename = "data.lmp"
            with io.open(os.path.join(raspa_path, in_file_rename), "w") as wf:
                # print("Writing input file: " + os.path.join(raspa_path, in_file_rename))
                with io.open(os.path.join(raspa_path, in_file_name), "r") as rf:
                    # print("Reading original input file: " + os.path.join(raspa_path, in_file_name))
                    wf.write(
                        rf.read().replace(
                            data_file_name,
                            data_file_rename))

            os.remove(os.path.join(raspa_path, in_file_name))
            shutil.move(
                os.path.join(
                    raspa_path, data_file_name), os.path.join(
                    raspa_path, data_file_rename))

            # print("Success!!\n\n")
            return_code = 0

        except Exception as e:
            print(e)
            # print("Failed!! Removing files...\n\n")
            shutil.rmtree(raspa_path)
            return_code = -1
            return raspa_path, return_code

        # print("Reading data file for element list: " + os.path.join(raspa_path, data_file_name))
        ff_style_dict = None
        with io.open(os.path.join(raspa_path, in_file_rename), "r") as rf1:
            ff_style_str = rf1.read()
            ff_style_list = list(
                filter(
                    None, [
                        x.strip() for x in ff_style_str.split("\n")]))
            ff_style_list = [
                pd.read_csv(
                    io.StringIO(x),
                    header=None,
                    sep=r"\s+").values.tolist()[0] for x in ff_style_list]
            ff_style_list = [[x[0], x[1:]] for x in ff_style_list]
            ff_style_dict = dict(zip(*list(map(list, zip(*ff_style_list)))))

        read_str = None
        with io.open(os.path.join(raspa_path, data_file_rename), "r") as rf2:
            read_str = rf2.read()
        mass_df = read_lmp_sec_str2df(read_str.split(
            "Masses")[1].split("Pair Coeffs")[0].strip())
        pair_coeff_df = read_lmp_sec_str2df(read_str.split(
            "Pair Coeffs")[1].split("Bond Coeffs")[0].strip())
        bond_coeff_df = read_lmp_sec_str2df(read_str.split(
            "Bond Coeffs")[1].split("Angle Coeffs")[0].strip())
        angle_coeff_df = read_lmp_sec_str2df(read_str.split(
            "Angle Coeffs")[1].split("Dihedral Coeffs")[0].strip())
        dihedral_coeff_df = read_lmp_sec_str2df(read_str.split("Dihedral Coeffs")[
                                                1].split("Improper Coeffs")[0].strip())
        improper_coeff_df = read_lmp_sec_str2df(
            read_str.split("Improper Coeffs")[1].split("Atoms")[0].strip())
        cifbox = read_str.split("Atoms")[1].split("$$$atoms$$$")[0].strip()
        atom_df = read_lmp_sec_str2df(read_str.split("$$$atoms$$$")[
                                      1].split("Bonds")[0].strip())
        atom_df.columns = [
            'id',
            'mol',
            'type',
            'q',
            'x',
            'y',
            'z',
            '#',
            "comment"]
        _atom_df = pd.read_csv(
            io.StringIO(
                "\n".join(
                    atom_df["comment"].to_list())),
            sep=r"\s+",
            header=None,
            index_col=None,
            names=[
                "comment",
                "fx",
                "fy",
                "fz"])
        atom_df = pd.concat([atom_df[['id', 'mol', 'type', 'q', 'x', 'y', 'z', '#']].reset_index(
            drop=True), _atom_df.reset_index(drop=True)], axis=1)
        bond_df = read_lmp_sec_str2df(
            read_str.split("Bonds")[1].split("Angles")[0].strip())
        angle_df = read_lmp_sec_str2df(read_str.split(
            "Angles")[1].split("Dihedrals")[0].strip())
        dihedral_df = read_lmp_sec_str2df(read_str.split(
            "Dihedrals")[1].split("Impropers")[0].strip())
        improper_df = read_lmp_sec_str2df(
            read_str.split("Impropers")[1].strip())

        cif_fname = self.rewrite_cif(raspa_path, cif_name, atom_df, cifbox)
        atom_fname = self.write_pseudo_atoms_def(
            raspa_path, ff_style_dict, mass_df, atom_df)
        mix_fname = self.write_force_field_mixing_rules_def(
            raspa_path, ff_style_dict, pair_coeff_df, atom_df)
        ff_fname = self.write_force_field_def(raspa_path)
        frm_fname = self.write_framework_def(
            raspa_path,
            bond_coeff_df,
            angle_coeff_df,
            dihedral_coeff_df,
            improper_coeff_df,
            bond_df,
            angle_df,
            dihedral_df,
            improper_df)

        # void fraction
        sim_dir = os.path.join(raspa_path, "helium_void_fraction")
        os.makedirs(sim_dir, exist_ok=True)
        shutil.copyfile(
            os.path.join(
                raspa_path, cif_fname), os.path.join(
                sim_dir, cif_fname))
        shutil.copyfile(
            os.path.join(
                raspa_path, atom_fname), os.path.join(
                sim_dir, atom_fname))
        shutil.copyfile(
            os.path.join(
                raspa_path, mix_fname), os.path.join(
                sim_dir, mix_fname))
        shutil.copyfile(
            os.path.join(
                raspa_path, ff_fname), os.path.join(
                sim_dir, ff_fname))
        shutil.copyfile(
            os.path.join(
                raspa_path, frm_fname), os.path.join(
                sim_dir, frm_fname))

        cell_rep = cell_replicate
        input_str = """SimulationType                       MonteCarlo
NumberOfCycles                       """ + "%d" % timesteps + """
PrintEvery                           """ + "%d" % report_frequency + """
PrintPropertiesEvery                 """ + "%d" % report_frequency + """

Forcefield                           local

Framework 0
FrameworkName mof
UnitCells """ + "%d" % cell_rep[0] + " " + "%d" % cell_rep[1] + " " + "%d" % cell_rep[2] + """
ExternalTemperature 298.0

Component 0 MoleculeName             helium
            MoleculeDefinition       local
            WidomProbability         1.0
            CreateNumberOfMolecules  0
"""
        with io.open(os.path.join(sim_dir, "simulation.input"), "w", newline="\n") as wf:
            wf.write(input_str)

        helium_str = """# critical constants: Temperature [T], Pressure [Pa], and Acentric factor [-]
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
# Chiral centers Bond  BondDipoles Bend  UrayBradley InvBend  Torsion Imp. Torsion""" + " " + \
            """Bond/Bond Stretch/Bend Bend/Bend Stretch/Torsion Bend/Torsion IntraVDW IntraCoulomb
               0    0            0    0            0       0        0            0""" + "         " + \
            """0            0         0               0            0        0            0
# Number of config moves
0
"""
        with io.open(os.path.join(sim_dir, "helium.def"), "w", newline="\n") as wf:
            wf.write(helium_str)

        # run void fraction now!

        # args = raspa_abs_path + " simulation.input"
        # task = subprocess.Popen(args,
        #                         cwd=os.path.abspath(sim_dir),
        #                         stdout=subprocess.PIPE,
        #                         stderr=subprocess.STDOUT)

        # read void fraction:
        # outdir = os.path.join(sim_dir, "Output")
        # outdir = os.path.join(outdir, os.listdir(outdir)[0])
        # outfile = os.path.join(outdir, [x for x in os.listdir(outdir) if "2.2.2" in x][0])
        # outstr = None
        # with io.open(outfile, "r", newline="\n") as rf:
        #     outstr = rf.read()
        # He_Void_Faction = float(outstr.split("[helium] Average Widom Rosenbluth-weight:")[1].split("+/-")[0])
        He_Void_Faction = 0.0

        # GCMC rigid!!!
        sim_dir = os.path.join(raspa_path, "co2_adsorption_rigid")
        os.makedirs(sim_dir, exist_ok=True)
        shutil.copyfile(
            os.path.join(
                raspa_path, atom_fname), os.path.join(
                sim_dir, atom_fname))
        shutil.copyfile(
            os.path.join(
                raspa_path, mix_fname), os.path.join(
                sim_dir, mix_fname))
        shutil.copyfile(
            os.path.join(
                raspa_path, ff_fname), os.path.join(
                sim_dir, ff_fname))
        shutil.copyfile(
            os.path.join(
                raspa_path, frm_fname), os.path.join(
                sim_dir, frm_fname))
        shutil.copyfile(cif_path, os.path.join(sim_dir, "mof.cif"))

        cell_rep = cell_replicate
        input_str = """SimulationType                MonteCarlo
NumberOfCycles                100000
NumberOfInitializationCycles  10000
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
FrameworkName mof
UnitCells """ + "%d" % cell_rep[0] + " " + "%d" % cell_rep[1] + " " + "%d" % cell_rep[2] + """
HeliumVoidFraction """ + "%1.4f" % He_Void_Faction + """
ExternalTemperature 298.15
ExternalPressure 1e4

Component 0 MoleculeName              CO2
            MoleculeDefinition        local
            IdealGasRosenbluthWeight  1.0
            TranslationProbability    1.0
            RotationProbability       1.0
            ReinsertionProbability    1.0
            SwapProbability           1.0
            CreateNumberOfMolecules   0

"""
        with io.open(os.path.join(sim_dir, "simulation.input"), "w", newline="\n") as wf:
            wf.write(input_str)

        co2_str = """# critical constants: Temperature [T], Pressure [Pa], and Acentric factor [-]
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
# Chiral centers Bond  BondDipoles Bend  UrayBradley InvBend  Torsion Imp. Torsion Bond/Bond""" + " " + \
            """Stretch/Bend Bend/Bend Stretch/Torsion Bend/Torsion IntraVDW IntraCoulomb
               0    2            0    0            0       0        0            0         0""" + "            " + \
            """0         0               0            0        0            0
# Bond stretch: atom n1-n2, type, parameters
0 1 RIGID_BOND
1 2 RIGID_BOND
# Number of config moves
0
"""
        with io.open(os.path.join(sim_dir, "CO2.def"), "w", newline="\n") as wf:
            wf.write(co2_str)

        slurm_str = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbvf-delta-cpu
#SBATCH --job-name=""" + sim_dir + """
#SBATCH --time=48:00:00      # hh:mm:ss for the job
#SBATCH -o delta_slurm-%j.log
#SBATCH -e delta_slurm-%j.log
#SBATCH --constraint="scratch&projects"
#SBATCH --mail-user=xyan11@uic.edu
#SBATCH --mail-type="BEGIN,END" # See sbatch or srun man pages for more email options
echo $SLURM_JOBID > jobid
valhost=$SLURM_JOB_NODELIST
echo $valhost > hostname
module purge
module load anaconda3_cpu/23.3.1
module list
source /sw/external/python/anaconda3-2023.03_cpu/etc/profile.d/conda.sh
lscpu
free -h
conda activate /projects/bbke/xyan11/conda-envs/gcmc2-310
ulimit -s unlimited
export RASPA_DIR=/projects/bbke/xyan11/conda-envs/gcmc2-310
cd """ + os.path.join("/scratch/bbvf/xyan11/gcmc/co2_0.1bar", sim_dir) + """
pwd
""" + raspa_abs_path + """ simulation.input"""
        with io.open(os.path.join(sim_dir, "run_gcmc.sbatch"), "w", newline="\n") as wf:
            wf.write(slurm_str)

        # GCMC!!!
        sim_dir = os.path.join(raspa_path, "co2_adsorption_flex")
        os.makedirs(sim_dir, exist_ok=True)
        shutil.copyfile(
            os.path.join(
                raspa_path, atom_fname), os.path.join(
                sim_dir, atom_fname))
        shutil.copyfile(
            os.path.join(
                raspa_path, mix_fname), os.path.join(
                sim_dir, mix_fname))
        shutil.copyfile(
            os.path.join(
                raspa_path, ff_fname), os.path.join(
                sim_dir, ff_fname))
        shutil.copyfile(
            os.path.join(
                raspa_path, frm_fname), os.path.join(
                sim_dir, frm_fname))
        shutil.copyfile(cif_path, os.path.join(sim_dir, "mof.cif"))

        cell_rep = [2, 2, 2]
        input_str = """SimulationType                MonteCarlo
NumberOfCycles                100000
NumberOfInitializationCycles  10000
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
FrameworkName mof
UnitCells """ + "%d" % cell_rep[0] + " " + "%d" % cell_rep[1] + " " + "%d" % cell_rep[2] + """
HeliumVoidFraction """ + "%1.4f" % He_Void_Faction + """
ExternalTemperature 298.15
ExternalPressure 1e4

FrameworkDefinitions local
FlexibleFramework yes

HybridNVEMoveProbability 1.0
  NumberOfHybridNVESteps 5

Component 0 MoleculeName              CO2
            MoleculeDefinition        local
            IdealGasRosenbluthWeight  1.0
            TranslationProbability    1.0
            RotationProbability       1.0
            ReinsertionProbability    1.0
            SwapProbability           1.0
            CreateNumberOfMolecules   0

"""
        with io.open(os.path.join(sim_dir, "simulation.input"), "w", newline="\n") as wf:
            wf.write(input_str)

        co2_str = """# critical constants: Temperature [T], Pressure [Pa], and Acentric factor [-]
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
# Chiral centers Bond  BondDipoles Bend  UrayBradley InvBend  Torsion Imp. Torsion Bond/Bond""" + " " + \
            """Stretch/Bend Bend/Bend Stretch/Torsion Bend/Torsion IntraVDW IntraCoulomb
               0    2            0    0            0       0        0            0         0""" + "            " + \
            """0         0               0            0        0            0
# Bond stretch: atom n1-n2, type, parameters
0 1 RIGID_BOND
1 2 RIGID_BOND
# Number of config moves
0
"""
        with io.open(os.path.join(sim_dir, "CO2.def"), "w", newline="\n") as wf:
            wf.write(co2_str)

        slurm_str = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbvf-delta-cpu
#SBATCH --job-name=""" + sim_dir + """
#SBATCH --time=48:00:00      # hh:mm:ss for the job
#SBATCH -o delta_slurm-%j.log
#SBATCH -e delta_slurm-%j.log
#SBATCH --constraint="scratch&projects"
#SBATCH --mail-user=xyan11@uic.edu
#SBATCH --mail-type="BEGIN,END" # See sbatch or srun man pages for more email options
echo $SLURM_JOBID > jobid
valhost=$SLURM_JOB_NODELIST
echo $valhost > hostname
module purge
module load anaconda3_cpu/23.3.1
module list
source /sw/external/python/anaconda3-2023.03_cpu/etc/profile.d/conda.sh
lscpu
free -h
conda activate /projects/bbke/xyan11/conda-envs/gcmc2-310
ulimit -s unlimited
export RASPA_DIR=/projects/bbke/xyan11/conda-envs/gcmc2-310
cd """ + os.path.join("/scratch/bbvf/xyan11/gcmc/co2_0.1bar", sim_dir) + """
pwd
""" + raspa_abs_path + """ simulation.input"""
        with io.open(os.path.join(sim_dir, "run_gcmc.sbatch"), "w", newline="\n") as wf:
            wf.write(slurm_str)

        return raspa_path, return_code
