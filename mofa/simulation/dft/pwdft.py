"""Run computations backed by PWDFT. Copied from Raymundo Hernandez and Alvaro Vazquez-Mayagoitia's implementation."""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import os

from ase import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.calculators.calculator import FileIOCalculator, KPoints, kpts2kpts, Calculator

from mofa.simulation.dft.base import BaseDFTRunner
import re
import json
import numpy as np
from copy import deepcopy

_special_kws = [
    "center",
    "autosym",
    "autoz",
    "theory",
    "basis",
    "xc",
    "task",
    "set",
    "symmetry",
    "label",
    "geompar",
    "basispar",
    "kpts",
    "bandpath",
    "restart_kw",
]

_system_type = {1: "polymer", 2: "surface", 3: "crystal"}


def _get_geom(atoms, **params):
    geom_header = ["geometry units angstrom"]
    for geomkw in ["center", "autosym", "autoz"]:
        geom_header.append(geomkw if params.get(geomkw) else "no" + geomkw)
    if "geompar" in params:
        geom_header.append(params["geompar"])
    geom = [" ".join(geom_header)]

    outpos = atoms.get_positions()
    pbc = atoms.pbc
    pbc[2] = True  # FIX IT
    if np.any(pbc):
        scpos = atoms.get_scaled_positions()
        for i, pbci in enumerate(pbc):
            if pbci:
                outpos[:, i] = scpos[:, i]
        npbc = pbc.sum()
        cellpars = atoms.cell.cellpar()
        geom.append("  system {} units angstrom".format(_system_type[npbc]))
        if npbc == 3:
            geom.append("    lattice_vectors")
            for row in atoms.cell:
                geom.append("      {:20.16e} {:20.16e} {:20.16e}".format(*row))
        else:
            if pbc[0]:
                geom.append("    lat_a {:20.16e}".format(cellpars[0]))
            if pbc[1]:
                geom.append("    lat_b {:20.16e}".format(cellpars[1]))
            if pbc[2]:
                geom.append("    lat_c {:20.16e}".format(cellpars[2]))
            if pbc[1] and pbc[2]:
                geom.append("    alpha {:20.16e}".format(cellpars[3]))
            if pbc[0] and pbc[2]:
                geom.append("    beta {:20.16e}".format(cellpars[4]))
            if pbc[1] and pbc[0]:
                geom.append("    gamma {:20.16e}".format(cellpars[5]))
        geom.append("  end")

    for i, atom in enumerate(atoms):
        geom.append(
            "  {:<2} {:20.16e} {:20.16e} {:20.16e}".format(atom.symbol, *outpos[i])
        )
    symm = params.get("symmetry")
    if symm is not None:
        geom.append("  symmetry {}".format(symm))
    geom.append("end")
    return geom


_special_keypairs = [
    ("nwpw", "simulation_cell"),
    ("nwpw", "carr-parinello"),
    ("nwpw", "brillouin_zone"),
    ("tddft", "grad"),
]


def _format_brillouin_zone(array, name=None):
    out = ["  brillouin_zone"]
    if name is not None:
        out += ["    zone_name {}".format(name)]
    template = "    kvector" + " {:20.16e}" * array.shape[1]
    for row in array:
        out.append(template.format(*row))
    out.append("  end")
    return out


def _get_bandpath(bp):
    if bp is None:
        return []
    out = ["nwpw"]
    out += _format_brillouin_zone(bp.kpts, name=bp.path)
    out += ["  zone_structure_name {}".format(bp.path), "end", "task band structure"]
    return out


def _format_line(key, val):
    if val is None:
        return key
    if isinstance(val, bool):
        return "{} .{}.".format(key, str(val).lower())
    else:
        return " ".join([key, str(val)])


def _format_block(key, val, twod_hcurve, lmbfgs, nindent=0):
    prefix = "  " * nindent
    prefix2 = "  " * (nindent + 1)
    if val is None:
        return [prefix + key]

    if not isinstance(val, dict):
        return [prefix + _format_line(key, val)]

    out = [prefix + key]
    if key == "nwpw" and twod_hcurve is True:
        out.append(prefix2 + "2d-hcurve")
    if key == "nwpw" and lmbfgs is True:
        out.append(prefix2 + "lmbfgs")
    for subkey, subval in val.items():
        if (key, subkey) in _special_keypairs:
            if (key, subkey) == ("nwpw", "brillouin_zone"):
                out += _format_brillouin_zone(subval)
            else:
                out += _format_block(subkey, subval, nindent=nindent + 1)
        else:
            if isinstance(subval, dict):
                subval = " ".join([_format_line(a, b) for a, b in subval.items()])
            out.append(prefix2 + " ".join([_format_line(subkey, subval)]))
    out.append(prefix + "end")
    return out


def _get_other(twod_hcurve, lmbfgs, **params):
    out = []

    for kw, block in params.items():
        if kw in _special_kws:
            continue
        out += _format_block(kw, block, twod_hcurve, lmbfgs)
    return out


def _get_set(**params):
    return ["set " + _format_line(key, val) for key, val in params.items()]


_gto_theories = ["tce", "ccsd", "mp2", "tddft", "scf", "dft"]
_pw_theories = ["band", "pspw", "paw"]
_all_theories = _gto_theories + _pw_theories


def _get_theory(**params):
    # Default: user-provided theory
    theory = params.get("theory")
    if theory is not None:
        return theory

    # Check if the user passed a theory to xc
    xc = params.get("xc")
    if xc in _all_theories:
        return xc

    # Check for input blocks that correspond to a particular level of
    # theory. Correlated theories (e.g. CCSD) are checked first.
    for kw in _gto_theories:
        if kw in params:
            return kw

    # If the user passed an 'nwpw' block, then they want a plane-wave
    # calculation, but what kind? If they request k-points, then
    # they want 'band', otherwise assume 'pspw' (if the user wants
    # to use 'paw', they will have to ask for it specifically).
    nwpw = params.get("nwpw")
    if nwpw is not None:
        if "monkhorst-pack" in nwpw or "brillouin_zone" in nwpw:
            return "band"
        return "pspw"

    # When all else fails, default to dft.
    return "dft"


_xc_conv = dict(
    lda="slater pw91lda",
    pbe="xpbe96 cpbe96",
    revpbe="revpbe cpbe96",
    rpbe="rpbe cpbe96",
    pw91="xperdew91 perdew91",
)


def _update_mult(magmom_tot, **params):
    theory = params["theory"]
    if magmom_tot == 0:
        magmom_mult = 1
    else:
        magmom_mult = np.sign(magmom_tot) * (abs(magmom_tot) + 1)
    if "scf" in params:
        for kw in [
            "nopen",
            "singlet",
            "doublet",
            "triplet",
            "quartet",
            "quintet",
            "sextet",
            "septet",
            "octet",
        ]:
            if kw in params["scf"]:
                break
        else:
            params["scf"]["nopen"] = magmom_tot
    elif theory in ["scf", "mp2", "ccsd", "tce"]:
        params["scf"] = dict(nopen=magmom_tot)

    if "dft" in params:
        if "mult" not in params["dft"]:
            params["dft"]["mult"] = magmom_mult
    elif theory in ["dft", "tddft"]:
        params["dft"] = dict(mult=magmom_mult)

    if "nwpw" in params:
        if "mult" not in params["nwpw"]:
            params["nwpw"]["mult"] = magmom_mult
    elif theory in ["pspw", "band", "paw"]:
        params["nwpw"] = dict(mult=magmom_mult)

    return params


def _get_kpts(atoms, **params):
    """Converts top-level 'kpts' argument to native keywords"""
    kpts = params.get("kpts")
    if kpts is None:
        return params

    nwpw = params.get("nwpw", dict())

    if "monkhorst-pack" in nwpw or "brillouin_zone" in nwpw:
        raise ValueError("Redundant k-points specified!")

    if isinstance(kpts, KPoints):
        nwpw["brillouin_zone"] = kpts.kpts
    elif isinstance(kpts, dict):
        if kpts.get("gamma", False) or "size" not in kpts:
            nwpw["brillouin_zone"] = kpts2kpts(kpts, atoms).kpts
        else:
            nwpw["monkhorst-pack"] = " ".join(map(str, kpts["size"]))
    elif isinstance(kpts, np.ndarray):
        nwpw["brillouin_zone"] = kpts
    else:
        nwpw["monkhorst-pack"] = " ".join(map(str, kpts))

    params["nwpw"] = nwpw
    return params


def write_pwdft_in(
        fd, atoms, properties=None, echo=True, twod_hcurve=True, lmbfgs=True, **params
):
    """Writes PWDFT input file.

    Parameters
    ----------
    fd
        file descriptor
    atoms
        atomic configuration
    properties
        list of properties to compute; by default only the
        calculation of the energy is requested
    echo
        if True include the `echo` keyword at the top of the file,
        which causes the content of the input file to be included
        in the output file
    params
        dict of instructions blocks to be included
    """
    params = deepcopy(params)

    if properties is None:
        properties = ["energy"]

    if "stress" in properties:
        if "set" not in params:
            params["set"] = dict()
        params["set"]["includestress"] = True

    task = params.get("task")
    if task is None:
        if "stress" in properties or "forces" in properties:
            task = "gradient"
        else:
            task = "energy"

    params = _get_kpts(atoms, **params)

    theory = _get_theory(**params)
    params["theory"] = theory
    xc = params.get("xc")
    if "xc" in params:
        xc = _xc_conv.get(params["xc"].lower(), params["xc"])
        if theory in ["dft", "tddft"]:
            if "dft" not in params:
                params["dft"] = dict()
            params["dft"]["xc"] = xc
        elif theory in ["pspw", "band", "paw"]:
            if "nwpw" not in params:
                params["nwpw"] = dict()
            params["nwpw"]["xc"] = xc

    magmom_tot = int(atoms.get_initial_magnetic_moments().sum())
    params = _update_mult(magmom_tot, **params)

    label = params.get("label", "pwdft")
    perm = os.path.abspath(params.pop("perm", label))
    scratch = os.path.abspath(params.pop("scratch", label))
    restart_kw = params.get("restart_kw", "start")
    if restart_kw not in ("start", "restart"):
        raise ValueError("Unrecognised restart keyword: {}!".format(restart_kw))
    short_label = label.rsplit("/", 1)[-1]
    if echo:
        out = ["echo"]
    else:
        out = []

    perm = "./perm"
    scratch = "./perm"

    out.extend(
        [
            'title "{}"'.format(short_label),
            "permanent_dir {}".format(perm),
            "scratch_dir {}".format(scratch),
            "{} {}".format(restart_kw, short_label),
            "\n".join(_get_geom(atoms, **params)),
            "\n".join(_get_other(twod_hcurve, lmbfgs, **params)),
            "\n".join(_get_set(**params.get("set", dict()))),
            "task {} {}".format(theory, task),
            "\n".join(_get_bandpath(params.get("bandpath", None))),
        ]
    )

    fd.write("\n\n".join(out))

    # TODO (wardlt): Do a better job at this, don't hard code it for every run
    fd.write("""
nwpw
   dplot
      density total valence_density.cube
   end
end

task pspw pspw_dplot
""")


def read_pwdft_out(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

    matches = []
    for idx, line in enumerate(lines):
        if re.findall(r">>> job completed", line):
            matches.append(idx)

    last_match = matches[-1]
    line = lines[last_match + 2]

    if "Next rtdbstr" in line:
        json_str = line.split("Next rtdbstr=")[-1]

        data = json.loads(json_str)

        pspw = data["pspw"]
        energy = pspw["energy"] * Hartree

        geo = data["geometries"]
        geo1 = geo["geometry"]
        nion = geo1["nion"]

        symbols = geo1["symbols"]
        cell = np.array(geo1["unita"]).reshape(3, 3)
        nwpw = data["nwpw"]
        dipole = np.array(nwpw["dipole"])
        coors = np.array(geo1["coords"]).reshape(nion, 3)

        atoms = Atoms(symbols, positions=coors, cell=cell)

        forces = np.zeros((nion, 3))

        if "fion" in pspw:
            forces = np.array(pspw["fion"]).reshape(nion, 3)
        forces *= Hartree / Bohr

        calc = SinglePointDFTCalculator(
            atoms=atoms,
            energy=energy,
            free_energy=energy,  # XXX Is this right?
            forces=forces,
            dipole=dipole,
            # quadrupole=quadrupole,
        )
        calc.kpts = 1
        atoms.calc = calc
        return atoms
    else:
        pass


class PWDFT(FileIOCalculator):
    """Class for doing PWDFT calculations.

    calc = PWDFT(label='pwdft', xc='LDA', ecut=70)
    """

    implemented_properties = ["energy", "forces", "stress"]
    command = "pwdft < PREFIX.nwxi > PREFIX.nwxo"
    accepts_bandpath_keyword = True  # To Check
    discard_results_on_any_change = True  # To Check
    echo = True
    twodhcurve = True

    default_parameters = dict()

    def __init__(
            self,
            restart=None,
            ignore_bad_restart_file=FileIOCalculator._deprecated,
            label="pwdft",
            atoms=None,
            **kwargs,
    ):
        """Construct PWDFT-Calculator object"""

        FileIOCalculator.__init__(
            self, restart, ignore_bad_restart_file, label, atoms, **kwargs
        )
        self.calc = None

    def write_input(self, atoms, properties, system_changes):
        """Write input parameters to files"""
        # Prepare perm and scratch directories
        perm = os.path.abspath(self.parameters.get("perm", "perm"))
        scratch = os.path.abspath(self.parameters.get("scratch", "perm"))
        os.makedirs(perm, exist_ok=True)
        os.makedirs(scratch, exist_ok=True)

        with open(self.label + ".nwxi", "w") as fd:
            write_pwdft_in(fd, atoms, properties, **self.parameters)

    def read_results(self):
        output = read_pwdft_out(self.label + ".nwxo")
        self.calc = output.calc
        self.results = output.calc.results


_pwdft_options = {
    "default": {
        "echo": True,
        "charge": 0,
        "nwpw": {"cutoff": 30, "xc": "PBE", "loop": "10 250"},
    }
}


@dataclass(kw_only=True)
class PWDFTRunner(BaseDFTRunner):
    """Interface for running pre-defined PWDFT workflows"""

    run_dir: Path = Path("pwdft-runs")
    """Path in which to store PWDFT computation files"""
    dft_cmd: str = 'mpiexec -n 1 pwdft'

    @contextmanager
    def _make_calc(self, level: str, out_dir: Path) -> Calculator:
        if level not in _pwdft_options:
            raise ValueError(f"No presets for {level}")
        options = _pwdft_options[level].copy()

        options["command"] = self.dft_cmd + " < PREFIX.nwxi > PREFIX.nwxo"

        # Make the perm directory
        #  TODO (wardlt): For some reason PWDFT cannot make directories?
        out_dir.joinpath('perm').mkdir(parents=True, exist_ok=True)

        start_dir = Path.cwd().absolute()
        try:
            os.chdir(out_dir)
            yield PWDFT(label='mof', **options)
        finally:
            os.chdir(start_dir)
