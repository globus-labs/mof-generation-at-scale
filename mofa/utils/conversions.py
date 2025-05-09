"""Functions useful for converting between data types"""
from io import StringIO

from ase import Atoms, io
from ase.calculators.singlepoint import SinglePointCalculator


def write_to_string(atoms: Atoms, fmt: str, **kwargs) -> str:
    """Write an ASE atoms object to string

    Args:
        atoms: Structure to write
        fmt: Target format
        kwargs: Passed to the write function
    Returns:
        Structure written in target format
    """

    out = StringIO()
    atoms.write(out, fmt, **kwargs)
    return out.getvalue()


def read_from_string(atoms_msg: str, fmt: str) -> Atoms:
    """Read an ASE atoms object from a string

    Args:
        atoms_msg: String format of the object to read
        fmt: Format (cannot be autodetected)
    Returns:
        Parsed atoms object
    """

    out = StringIO(str(atoms_msg))  # str() ensures that Proxies are resolved
    return io.read(out, format=fmt)


def canonicalize(atoms: Atoms) -> Atoms:
    """Replace the calculator in an Atoms with a serializable version

    Args:
        atoms: Structure to write
    Returns:
        Atoms object that has been serialized and deserialized
    """
    # TODO (wardlt): Make it so the single-point calculator can hold unknown properties? (see discussion https://gitlab.com/ase/ase/-/issues/782)
    # Borrowed from https://github.com/globus-labs/cascade/blob/main/cascade/utils.py
    out_atoms = atoms.copy()
    if atoms.calc is not None:
        old_calc = atoms.calc
        out_atoms.calc = SinglePointCalculator(atoms)
        out_atoms.calc.results = old_calc.results.copy()
    return out_atoms
