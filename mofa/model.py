"""Data models for a MOF class"""
from dataclasses import dataclass, field, asdict
from functools import cached_property
from hashlib import sha512
from pathlib import Path
from io import StringIO
from uuid import uuid4
import json

import yaml
import numpy as np
from ase.io import read
from ase.io.vasp import read_vasp
import ase

from mofa.utils.conversions import read_from_string, write_to_string

import pandas as pd
import itertools
import io


@dataclass
class NodeDescription:
    """The inorganic components of a MOF"""

    smiles: str = field()
    """SMILES-format description of the node"""
    xyz: str | None = field(default=None, repr=False)
    """XYZ coordinates of each atom in the node

    Uses At or Fr as an identifier of the the anchor points
    where the linkers attach to the node
    - At designates a carbon-carbon bond anchor
    - Fr designates other types of linkages
    """


@dataclass
class LigandTemplate:
    """Beginning of a new ligand to be generated.

    Contains only the proper end groups oriented in the sizes needed by our MOF."""

    role: str
    """Portion of the MOF to which this ligand corresponds"""
    xyzs: tuple[str]
    """XYZ coordinates of the anchor groups"""
    dummy_element: str
    """Dummy element used to replace end group when assembling MOF"""

    @cached_property
    def anchors(self) -> list[ase.Atoms]:
        """The anchor groups as ASE objects"""
        return [read_from_string(xyz, 'xyz') for xyz in self.xyzs]

    def prepare_inputs(self) -> tuple[list[str], np.ndarray]:
        """Produce the inputs needed for DiffLinker

        Returns:
            - List of chemical symbols
            - Array of atomic positions
        """
        symbols = []
        positions = []
        for xyz in self.xyzs:
            atoms = read_from_string(xyz, fmt='xyz')
            symbols.extend(atoms.get_chemical_symbols())
            positions.append(atoms.positions)
        return symbols, np.concatenate(positions, axis=0)

    def create_description(self, atom_types: list[str], coordinates: np.ndarray) -> 'LigandDescription':
        """Produce a ligand description given atomic coordinates which include the infilled atoms

        Args:
            atom_types: Types of all atoms as chemical symbols
            coordinates: Coordinates of all atoms
        Returns:
            Ligand description using the new coordinates
        """

        # The linker groups should be up from, make sure the types have not changed and change if they have
        pos = 0
        anchor_atoms = []
        for anchor in self.anchors:
            # Determine the fragment positions
            found_types = atom_types[pos:pos + len(anchor)]
            anchor_atoms.append(list(range(pos, pos + len(anchor))))
            expected_types = anchor.get_chemical_symbols()

            # Make sure the types match, and increment position
            assert found_types == expected_types, f'Anchor changed. Found: {found_types} - Expected: {expected_types}'
            pos += len(anchor)

        # Build the XYZ file
        atoms = ase.Atoms(symbols=atom_types, positions=coordinates)
        return LigandDescription(
            xyz=write_to_string(atoms, 'xyz'),
            role=self.role,
            anchor_atoms=anchor_atoms,
            dummy_element=self.dummy_element
        )

    @classmethod
    def from_yaml(cls, path: Path) -> 'LigandTemplate':
        """Load a template description from YAML

        Args:
            path: Path to the YAML file
        Returns:
            The ligand template
        """

        with path.open() as fp:
            return cls(**yaml.safe_load(fp))



def bulkRemoveAtoms(emol, atoms2rm):
    emol_copy = RWMol(emol)
    for a2rm in atoms2rm:
        emol_copy.GetAtomWithIdx(a2rm).SetAtomicNum(0)
    emol_copy = Chem.DeleteSubstructs(emol_copy, Chem.MolFromSmarts('[#0]'))
    emol = RWMol(emol_copy)
    return emol

def bulkRemoveBonds(emol, bonds2rm, fragAllowed=False):
    emol_copy = RWMol(emol)
    for b2rm in bonds2rm:
        _emol_copy = RWMol(emol_copy)
        _emol_copy.RemoveBond(b2rm[0], b2rm[1])
        if not fragAllowed:
            if len(GetMolFrags(_emol_copy)) == 1:
                emol_copy = RWMol(_emol_copy)
        else:
            emol_copy = RWMol(_emol_copy)
    emol = RWMol(emol_copy)
    return emol

def rdkitGetLargestCC(emol):
    GetMolFrags(emol)
    atoms2rm = list(
        itertools.chain(
            *
            sorted(
                GetMolFrags(emol),
                key=lambda x: len(x),
                reverse=True)[
                1:]))
    emol = bulkRemoveAtoms(emol, atoms2rm)
    return emol


@dataclass
class LigandDescription:
    """Description of organic sections which connect inorganic nodes"""

    smiles: str | None = field(default=None)
    """SMILES-format designation of the molecule"""
    xyz: str | None = field(default=None, repr=False)
    """XYZ coordinates of each atom in the linker"""
    role: str | None = field(default=None)
    """Portion of the MOF to which this ligand corresponds"""

    anchor_atoms: list[list[int]] | None = field(default=None, repr=True)
    """Groups of atoms which attach to the nodes

    There are typically two groups of fragment atoms, and these are
    never altered during MOF generation."""
    dummy_element: str = field(default=None, repr=False)
    """Element used to represent the end group during assembly"""



    @cached_property
    def atoms(self):
        return read_from_string(self.xyz, "xyz")

    def replace_with_dummy_atoms(self, anchor_types: str="COO") -> ase.Atoms:
        """Replace the fragments which attach to nodes with dummy atoms"""
        
        df = pd.read_csv(io.StringIO(li.xyz), skiprows=2, sep=r"\s+", header=None, names=["element", "x", "y", "z"])
        anchor_ids = list(itertools.chain(*li.anchor_atoms))
        anchor_df = df.loc[anchor_ids, :]
        at_id = anchor_df[anchor_df["element"]=="C"].index
        remove_ids = anchor_df[anchor_df["element"]=="O"].index
        df.loc[at_id, "element"] = "At"
        df = df.loc[list(set(df.index)-set(remove_ids)), :]
        return ase.Atoms(df["element"], df.loc[:, ["x", "y", "z"]].values)

    @classmethod
    def from_yaml(cls, path: Path) -> 'LigandDescription':
        """Load a ligand description from YAML

        Args:
            path: Path to the YAML file
        Returns:
            The ligand description
        """
        with path.open() as fp:
            return cls(**yaml.safe_load(fp))


@dataclass
class MOFRecord:
    """Information available about a certain MOF"""
    # Data describing what the MOF is
    name: str = None
    """Name to be used for output files associated with this MOFs"""
    identifiers: dict[str, str] = field(default_factory=dict)
    """Names of this MOFs is registries (e.g., hMOF)"""
    topology: str | None = None
    """Description of the 3D network structure (e.g., pcu) as the topology"""
    catenation: int | None = None
    """Degree of catenation. 0 corresponds to no interpenetrating lattices"""
    nodes: tuple[NodeDescription, ...] = field(default_factory=tuple)
    """Description of the nodes within the structure"""
    ligands: tuple[LigandDescription, ...] = field(default_factory=tuple)
    """Description of each linker within the structure"""

    # Information about the 3D structure of the MOF
    structure: str | None = field(default=None, repr=False)
    """A representative 3D structure of the MOF in POSCAR format"""

    # Detailed outputs from simulations
    md_trajectory: dict[str, list[str]] = field(default_factory=dict, repr=False)
    """Structures of the molecule produced during molecular dynamics simulations.

    Key is the name of the level of accuracy for the MD computational (e.g., "uff"),
    values are the structure in POSCAR format"""

    # Properties
    gas_storage: dict[tuple[str, float], float] = field(default_factory=dict, repr=False)
    """Storage capacity of the MOF for different gases and pressures"""
    structure_stability: dict[str, float] = field(default_factory=dict, repr=False)
    """How likely the structure is to be stable according to different assays

    A score of 1 equates to most likely to be stable, 0 as least likely."""

    def __post_init__(self):
        if self.name is None:
            # Make a name by hashing
            hasher = sha512()
            if self.structure is not None:
                hasher.update(self.structure.encode())
            else:
                hasher.update(str(uuid4()).encode())

            self.name = f'mof-{hasher.hexdigest()[-8:]}'

    @classmethod
    def from_file(cls, path: Path | str, read_kwargs: dict | None = None, **kwargs) -> 'MOFRecord':
        """Create a MOF description from a structure file on disk

        The file will be read using ASE's ``read`` function.

        Keyword arguments can include identifiers of the MOF and
        will be passed to the constructor.

        Args:
            path: Path to the file
            read_kwargs: Arguments passed to the read function
        Returns:
            A MOF record before fragmentation
        """

        # Read the file from disk, then write to VASP
        if read_kwargs is None:
            read_kwargs = {}
        atoms = read(path, **read_kwargs)
        fp = StringIO()
        atoms.write(fp, format='vasp')

        return MOFRecord(structure=fp.getvalue(), **kwargs)

    @cached_property
    def atoms(self) -> ase.Atoms:
        """The structure as an ASE Atoms object"""
        return read_vasp(StringIO(self.structure))

    def to_json(self, **kwargs) -> str:
        """Render the structure to a JSON string

        Keyword arguments are passed to :meth:`json.dumps`

        Returns:
            JSON-format version of the object
        """

        return json.dumps(asdict(self), **kwargs)
