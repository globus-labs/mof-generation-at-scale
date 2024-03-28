"""Data models for a MOF class"""
from dataclasses import dataclass, field, asdict
from functools import cached_property
from datetime import datetime
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
import pandas as pd
import itertools
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, AllChem

from mofa.utils.conversions import read_from_string, write_to_string
from mofa.utils.xyz import unsaturated_xyz_to_xyz


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

    anchor_type: str
    """Type of anchoring group"""
    xyzs: tuple[str]
    """XYZ coordinates of the prompt fragments.

    Each XYZ must be arranged such that the first atom is the one which connects to the rest of the molecule,
    and the last set of atoms are those which are replaced with the dummy element."""
    dummy_element: str
    """Dummy element used to replace end group when assembling MOF"""

    @cached_property
    def anchors(self) -> list[ase.Atoms]:
        """The anchor groups as ASE objects"""
        return [read_from_string(xyz, 'xyz') for xyz in self.xyzs]

    def prepare_inputs(self) -> tuple[list[str], np.ndarray, np.ndarray | None]:
        """Produce the inputs needed for DiffLinker

        Returns:
            - List of chemical symbols
            - Array of atomic positions
            - Indices of the atom in each linker which connects to the rest of the molecule, if known
        """
        symbols = []
        positions = []
        start_ids = []
        for xyz in self.xyzs:
            # Mark the ID of the start atom
            start_ids.append(len(symbols))  # The next atom to be is the connecting atom

            # Add the atoms and positions to the outputs
            atoms = read_from_string(xyz, fmt='xyz')
            symbols.extend(atoms.get_chemical_symbols())
            positions.append(atoms.positions)

        return symbols, np.concatenate(positions, axis=0), start_ids

    def create_description(self, atom_types: list[str], coordinates: np.ndarray) -> 'LigandDescription':
        """Produce a ligand description given atomic coordinates which include the infilled atoms

        Assumes the provided coordinates are of the backbone atom and not the

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

        # Add Hydrogens to the molecule
        atoms = ase.Atoms(symbols=atom_types, positions=coordinates)
        unsat_xyz = write_to_string(atoms, 'xyz')
        sat_xyz = unsaturated_xyz_to_xyz(unsat_xyz, exclude_atoms=list(itertools.chain(*anchor_atoms)))

        return LigandDescription(
            xyz=sat_xyz,
            anchor_type=self.anchor_type,
            anchor_atoms=anchor_atoms,
            dummy_element=self.dummy_element
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> 'LigandTemplate':
        """Load a template description from YAML

        Args:
            path: Path to the YAML file
        Returns:
            The ligand template
        """

        with Path(path).open() as fp:
            return cls(**yaml.safe_load(fp))


@dataclass
class LigandDescription:
    """Description of organic sections which connect inorganic nodes"""

    smiles: str | None = field(default=None)
    """SMILES-format designation of the molecule"""
    xyz: str | None = field(default=None, repr=False)
    """XYZ coordinates of all atoms in the linker including all Hs"""
    sdf: str | None = field(default=None, repr=False)
    """SDF file string with atom positions and bond (with order) information (optional)"""

    # Information about how this ligand anchors to the inorganic portions
    anchor_type: str | None = field(default=None)
    """Name of the functional group used for anchoring"""
    anchor_atoms: list[list[int]] | None = field(default=None, repr=True)
    """Groups of atoms which attach to the nodes

    There are typically two groups of fragment atoms, and these are
    never altered during MOF generation."""
    dummy_element: str = field(default=None, repr=False)
    """Element used to represent the anchor group during assembly"""

    @cached_property
    def atoms(self):
        return read_from_string(self.xyz, "xyz")

    def anchor_constrained_optimization(self, xyz_tol=0.001, force_constant=10000.0, max_iterations=1000):
        """optimize the ligand while the anchor atoms are constrained

        Args:
            xyz_tol: coordinate tolerance for constrained atoms
            force_constant: the spring force constant to keep the constrained atoms in original position
            max_iterations: maximum number of iterations for optimization
        Returns:
            inplace function, no return
        """

        mol = Chem.MolFromXYZBlock(self.xyz)
        all_anchor_atoms = list(itertools.chain(*self.anchor_atoms))
        if self.anchor_type != "COO" and self.dummy_element != "At" and len(self.anchor_atoms[0]) != 3:
            charge = int(0)
        else:
            charge = int(-len(self.anchor_atoms))
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
        rdDetermineBonds.DetermineConnectivity(mol)
        rdDetermineBonds.DetermineBondOrders(mol)
        molprop = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, molprop)
        # set constraints
        for i in all_anchor_atoms:
            ff.MMFFAddPositionConstraint(i, xyz_tol, force_constant)
        ff.Minimize(maxIts=max_iterations)
        self.xyz = Chem.MolToXYZBlock(mol)

    def swap_cyano_with_COO(self):
        """create a new LigandDescription object with the same middle part but with the -COO instead of -cyano groups

        Returns:
            the new -COO LigandDescription object
        """

        if self.dummy_element != "Fr" or self.anchor_type == "COO":
            raise ValueError()

        # Generate the new XYZ file
        carboxylic_ion_CO_length = 1.26
        df = pd.read_csv(StringIO(self.xyz), skiprows=2, sep=r"\s+", header=None, names=["element", "x", "y", "z"])
        anchor_ids = list(itertools.chain(*self.anchor_atoms))
        other_atom_ids = list(set(df.index.tolist()) - set(anchor_ids))
        other_atom_df = df.loc[other_atom_ids, :]
        final_df_list = []
        new_anchor_ids = []
        for anc in self.anchor_atoms:
            anc = anc[-2:]  # Only the last two atoms are the C#N
            Cid = anc[0]
            Nid = anc[1]
            Cxyz = df.loc[Cid, ["x", "y", "z"]].values
            Nxyz = df.loc[Nid, ["x", "y", "z"]].values
            bisector = Cxyz - Nxyz
            bisector = bisector / np.linalg.norm(bisector)
            O1norm_vec = np.array([[0.5, -0.866, 0.], [0.866, 0.5, 0.], [0., 0., 1.]]) @ bisector
            O2norm_vec = np.array([[0.5, 0.866, 0.], [-0.866, 0.5, 0.], [0., 0., 1.]]) @ bisector
            O1xyz = -O1norm_vec * carboxylic_ion_CO_length + Cxyz
            O2xyz = -O2norm_vec * carboxylic_ion_CO_length + Cxyz
            new_COO_df = pd.DataFrame(np.array([Cxyz, O1xyz, O2xyz]), columns=["x", "y", "z"])
            new_COO_df["element"] = ["C", "O", "O"]
            new_COO_df = new_COO_df[["element", "x", "y", "z"]]
            final_df_list.append(new_COO_df)
        new_anchor_df = pd.concat(final_df_list, axis=0).reset_index(drop=True)
        flat_anchor_ids = np.array(new_anchor_df.index)
        new_anchor_ids = flat_anchor_ids.reshape(int(flat_anchor_ids.shape[0] / 3), 3).tolist()
        final_df = pd.concat([new_anchor_df, other_atom_df.copy(deep=True)], axis=0).reset_index(drop=True)
        new_xyz_str = str(len(final_df)) + "\n\n" + final_df.to_string(header=None, index=None)
        return LigandDescription(anchor_type="COO", xyz=new_xyz_str, anchor_atoms=new_anchor_ids, dummy_element="At")

    def replace_with_dummy_atoms(self) -> ase.Atoms:
        """Replace the fragments which attach to nodes with dummy atoms

        Returns:
            ASE atoms version of the molecule where the anchor atoms have been
            replaced with a single atom of the designated dummy type
        """

        # Get the locations of the
        output = read_from_string(self.xyz, 'xyz')

        # Each anchor type has different logic for where to place the dummy atom
        if self.anchor_type == "COO":
            # Place the dummy on the site of the carbon
            to_remove = []
            for curr_anchor in self.anchor_atoms:
                # Change the carbon atom's type
                symbols = output.get_chemical_symbols()
                at_id = curr_anchor[0]
                assert symbols[at_id] == 'C', 'The first anchor atom is not carbon'
                symbols[at_id] = self.dummy_element
                output.set_chemical_symbols(symbols)

                # Delete the other two atoms (both Oxygen)
                assert all(symbols[t] == 'O' for t in curr_anchor[1:]), 'The other anchors are not oxygen'
                to_remove.extend(curr_anchor[1:])

            del output[to_remove]
        elif self.anchor_type == "cyano":
            # Place the dummy atom between the C and N
            to_remove = []
            for curr_anchor in self.anchor_atoms:
                # Change the carbon atom's type
                symbols = output.get_chemical_symbols()
                at_id = curr_anchor[0]
                assert symbols[at_id] == 'C', 'The first anchor atom is not carbon'
                symbols[at_id] = self.dummy_element
                output.set_chemical_symbols(symbols)

                # Put the move the carbon atom towards the N
                output.positions[at_id, :] = output.positions[curr_anchor, :].mean(axis=0)

                # Delete the other two atoms (both Oxygen)
                assert symbols[curr_anchor[1]] == 'N'
                to_remove.append(curr_anchor[1])

            del output[to_remove]
        else:
            raise NotImplementedError(f'Logic not yet defined for anchor_type={self.anchor_type}')

        return output

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
    """Name to be used for output files associated with this MOFs. Assumed to be unique"""
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
    gas_storage: dict[str, tuple[float, float]] = field(default_factory=dict, repr=False)
    """Storage capacity of the MOF for different gases and pressures. Key is the name of the gas, value is the pressure and capacity (units TBD)"""
    structure_stability: dict[str, float] = field(default_factory=dict, repr=False)
    """How likely the structure is to be stable according to different assays

    A score of 1 equates to most likely to be stable, 0 as least likely."""

    # Tracking proveance of structure
    times: dict[str, datetime] = field(default_factory=lambda: {'created': datetime.now()})
    """Listing times at which key events occurred"""

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
