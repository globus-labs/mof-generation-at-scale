"""Data models for a MOF class"""
from dataclasses import dataclass, field, asdict
from functools import cached_property
from hashlib import sha512
from pathlib import Path
from io import StringIO
import json
from uuid import uuid4

from ase.io import read
from ase.io.vasp import read_vasp
import ase


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
class LigandDescription:
    """Description of organic sections which connect inorganic nodes"""

    smiles: str = field()
    """SMILES-format designation of the molecule"""
    xyz: str | None = field(default=None, repr=False)
    """XYZ coordinates of each atom in the linker"""

    fragment_atoms: list[list[int]] | None = field(default=None, repr=True)
    """Groups of atoms which attach to the nodes

    There are typically two groups of fragment atoms, and these are
    never altered during MOF generation."""

    @property
    def linker_atoms(self) -> list[int]:
        """All atoms which are not part of a fragment"""
        raise NotImplementedError()

    def generate_template(self, spacing_distance: float | None = None) -> ase.Atoms:
        """Generate a version of the ligand with only the fragments at the end

        Args:
            spacing_distance: Distance to enforce between the fragments. Set to ``None``
                to keep the current distance
        Returns:
            The template with the desired spacing
        """
        raise NotImplementedError()


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
    nodes: tuple[NodeDescription] = field(default_factory=tuple)
    """Description of the nodes within the structure"""
    ligands: tuple[LigandDescription] = field(default_factory=tuple)
    """Description of each linker within the structure"""

    # Information about the 3D structure of the MOF
    structure: str | None = field(default=None, repr=False)
    """A representative 3D structure of the MOF in POSCAR format"""

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
