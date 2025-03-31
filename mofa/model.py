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
import torch
import torch.nn.functional as F
import numpy as np
from ase import Atom
from ase.io import read
from ase.io.vasp import read_vasp
import ase
import itertools
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, AllChem

from mofa.utils.conversions import read_from_string, write_to_string
from mofa.utils.xyz import unsaturated_xyz_to_xyz
from mofa.utils.src import const


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

    Contains a prompt that includes both the anchor groups (i.e., those which connect to the node)
    and additional context atoms."""

    anchor_type: str
    """Type of anchoring group"""
    xyzs: tuple[str]
    """XYZ coordinates of the prompt fragments.

    Each XYZ must be arranged such that the first atom is the one which connects to the rest of the molecule,
    and the last set of atoms are those which are replaced with the dummy element."""
    dummy_element: str
    """Dummy element used to replace end group when assembling MOF"""

    @cached_property
    def prompts(self) -> list[ase.Atoms]:
        """The prompt fragments as ASE objects"""
        return [read_from_string(xyz, 'xyz') for xyz in self.xyzs]

    def prepare_inputs(self) -> tuple[list[str], np.ndarray, np.ndarray | None]:
        """Produce the inputs needed for DiffLinker

        Uses the prompts in :attr:`xyzs` without hte hydrogens

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
            is_not_h = [s != 'H' for s in atoms.get_chemical_symbols()]
            atoms = atoms[is_not_h]
            symbols.extend(atoms.get_chemical_symbols())
            positions.append(atoms.positions)

        return symbols, np.concatenate(positions, axis=0), start_ids

    def create_description(self, atom_types: list[str], coordinates: np.ndarray) -> 'LigandDescription':
        """Produce a ligand description given atomic coordinates which include the infilled atoms

        Assumes the provided coordinates are of the backbone atom and do not include Hydrogens.
        Adds in the hydrogens from the original prompt and infers them for the new symbols

        Args:
            atom_types: Types of all atoms as chemical symbols
            coordinates: Coordinates of all atoms
        Returns:
            Ligand description using the new coordinates
        """

        # Determine the coordinates belonging to the new atoms
        orig_symbols, orig_positions, _ = self.prepare_inputs()
        num_new_atoms = coordinates.shape[0] - orig_positions.shape[0]
        new_coordinates = coordinates[-num_new_atoms:, :]
        new_symbols = atom_types[-num_new_atoms:]

        # Make sure the supplied coordinates and symbols have not moved
        found_symbols = atom_types[:-num_new_atoms]
        assert orig_symbols == atom_types[:-num_new_atoms], f'Prompt changed. Found: {found_symbols} - Expected: {orig_symbols}'
        assert np.isclose(orig_positions, coordinates[:-num_new_atoms, :], atol=1e-2).all(), 'Coordinates have moved'

        # Build the new molecule by appending the new atoms to the end of the linkers
        atoms = self.prompts[0].copy()
        prompt_atoms = [list(range(len(atoms)))]
        pos = len(atoms)
        for prompt in self.prompts[1:]:
            atoms += prompt
            prompt_atoms.append(list(range(pos, pos + len(prompt))))
            pos += len(prompt)
        atoms += ase.Atoms(symbols=new_symbols, positions=new_coordinates)

        # Add Hydrogens to the molecule
        unsat_xyz = write_to_string(atoms, 'xyz')
        sat_xyz = unsaturated_xyz_to_xyz(unsat_xyz, exclude_atoms=list(itertools.chain(*prompt_atoms)))
        ld = LigandDescription(
            xyz=sat_xyz,
            anchor_type=self.anchor_type,
            prompt_atoms=prompt_atoms,
            dummy_element=self.dummy_element
        )
        try:
            ld.full_ligand_optimization()
        except (ValueError, AttributeError,):
            pass
        return ld

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

    name: str = None
    """Unique name for the ligand"""
    smiles: str | None = field(default=None)
    """SMILES-format designation of the molecule"""
    xyz: str | None = field(default=None, repr=False)
    """XYZ coordinates of all atoms in the linker including all Hs"""

    # Information about how this ligand prompts to the inorganic portions
    anchor_type: str | None = field(default=None)
    """Name of the functional group used for anchoring"""
    prompt_atoms: list[list[int]] | None = field(default=None, repr=True)
    """Groups of atoms which attach to the nodes

    There are typically two groups of fragment atoms, and these are
    never altered during MOF generation."""
    dummy_element: str = field(default=None, repr=False)
    """Element used to represent the anchor group during assembly"""

    metadata: dict[str, int | float | str] = field(default_factory=dict)
    """Any notable information about the molecule"""

    def __post_init__(self):
        if self.name is None:
            # Make a name by hashing
            hasher = sha512()
            if self.xyz is not None:
                hasher.update(self.xyz.encode())
            else:
                hasher.update(str(uuid4()).encode())

            self.name = f'ligand-{hasher.hexdigest()[-8:]}'

    @cached_property
    def atoms(self):
        return read_from_string(self.xyz, "xyz")

    def full_ligand_optimization(self, max_iterations=1000):
        """optimize the ligand while the anchor atoms are constrained

        Args:
            max_iterations: maximum number of iterations for optimization
        Returns:
            inplace function, no return
        """

        mol = Chem.MolFromXYZBlock(self.xyz)
        # all_anchor_atoms = list(itertools.chain(*self.prompt_atoms))
        charge = 0  # added hydrogen to COO ligand template, so no more charges
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
        rdDetermineBonds.DetermineConnectivity(mol)
        rdDetermineBonds.DetermineBondOrders(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iterations)
        self.xyz = Chem.MolToXYZBlock(mol)

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
        all_anchor_atoms = list(itertools.chain(*self.prompt_atoms))
        if self.anchor_type != "COO" and self.dummy_element != "At" and len(self.prompt_atoms[0]) != 3:
            charge = int(0)
        else:
            charge = int(-len(self.prompt_atoms))
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

    def replace_with_dummy_atoms(self) -> ase.Atoms:
        """Replace the fragments which attach to nodes with dummy atoms

        Returns:
            ASE atoms version of the molecule where the anchor atoms have been
            replaced with a single atom of the designated dummy type
        """

        # Get the locations of the atoms
        output = read_from_string(self.xyz, 'xyz')

        # Each anchor type has different logic for where to place the dummy atom
        if self.anchor_type == "COO":
            # Place the dummy on the site of the carbon
            to_remove = []
            for curr_anchor in self.prompt_atoms:
                curr_anchor = curr_anchor[-4:]  # changed from 3 to 4 to test -COOH instead of -COO
                # Change the carbon atom's type
                symbols = output.get_chemical_symbols()
                at_id = curr_anchor[0]
                assert symbols[at_id] == 'C', 'The first anchor atom is not carbon'
                symbols[at_id] = self.dummy_element
                output.set_chemical_symbols(symbols)

                # Delete the other two atoms (both Oxygen)
                assert all(symbols[t] in ['H', 'O'] for t in curr_anchor[1:]), 'The other prompts are not oxygen'
                to_remove.extend(curr_anchor[1:])

            del output[to_remove]
        elif self.anchor_type == "cyano":
            # Place the dummy atom 2A away from the N, along the direction of C#N bond
            for curr_anchor in self.prompt_atoms:
                curr_anchor = curr_anchor[-2:]
                # Check types
                symbols = output.get_chemical_symbols()
                assert symbols[curr_anchor[0]] == 'C'

                # Locate the new position
                c_pos = output.positions[curr_anchor[0], :]
                n_pos = output.positions[curr_anchor[1], :]
                vector_dir = n_pos - c_pos
                dummy_pos = n_pos + vector_dir / np.linalg.norm(vector_dir) * 2
                output.append(Atom(symbol=self.dummy_element, position=dummy_pos))
        elif self.anchor_type == "pyridine":
            # Place the dummy atom 2A away from the N, along the direction of C#N bond
            for curr_anchor in self.prompt_atoms:
                curr_anchor = curr_anchor[-2:]
                # Check types
                symbols = output.get_chemical_symbols()
                assert symbols[curr_anchor[0]] == 'C'

                # Locate the new position
                c_pos = output.positions[curr_anchor[0], :]
                n_pos = output.positions[curr_anchor[1], :]
                vector_dir = n_pos - c_pos
                dummy_pos = n_pos + vector_dir / np.linalg.norm(vector_dir) * 2
                output.append(Atom(symbol=self.dummy_element, position=dummy_pos))
        else:
            raise NotImplementedError(f'Logic not yet defined for anchor_type={self.anchor_type}')

        return output

    def to_training_example(self) -> dict:
        """Render this ligand into the training format for DiffLinker

        Returns:
            Difflinker-format representation of the training example
        """

        # Start with a unique name of the ligand
        training_ligand = {"uuid": str(uuid4()), "name": self.smiles}

        # Get the molecule without hydrogens, and a mapping between old and new index
        atoms_with_h = self.atoms.copy()
        original_ids = []
        for i, atom in enumerate(atoms_with_h):
            if atom.symbol != "H":
                original_ids.append(i)
        atoms_no_h = atoms_with_h[[a.symbol != "H" for a in atoms_with_h]]

        old_to_new_ind = dict((a, i) for i, a in enumerate(original_ids))

        # grab position information removing atom types and hydrogens
        positions = torch.tensor(atoms_no_h.positions, dtype=torch.float)
        training_ligand["positions"] = positions

        # get the number of atoms with hydrogens removed and create one-hot encoding of atom types
        atom_numbers = [
            const.GEOM_ATOM2IDX[atom] for atom in atoms_no_h.get_chemical_symbols()
        ]
        training_ligand["num_atoms"] = len(atoms_no_h)
        atom_numbers_tensor = torch.tensor(
            atom_numbers, dtype=torch.long
        )
        one_hot_encoded = F.one_hot(
            atom_numbers_tensor, num_classes=len(const.GEOM_ATOM2IDX.keys())
        )
        training_ligand["one_hot"] = one_hot_encoded.float()

        # convert atom types to charges
        charges = [
            const.GEOM_CHARGES[atom] for atom in atoms_no_h.get_chemical_symbols()
        ]
        training_ligand["charges"] = torch.tensor(
            charges, dtype=torch.float
        )

        # Map the IDs of prompt atoms and anchor atoms (first atom in prompt) to new indices (of the prompts w/o H)
        anchor_ids = set()
        prompt_ids = set()
        for prompt in self.prompt_atoms:
            anchor_ids.add(old_to_new_ind[prompt[0]])
            prompt_ids.update(old_to_new_ind[p] for p in prompt if p in old_to_new_ind)

        training_ligand["linker_mask"] = torch.tensor(
            [0 if i in prompt_ids else 1 for i in range(len(atoms_no_h))], dtype=torch.float
        )
        training_ligand["fragment_mask"] = 1 - training_ligand["linker_mask"]

        # create anchors for difflinker. Anchors are the first atom in the list of each prompt
        new_anchor_array = np.full(
            len(atoms_no_h), fill_value=False, dtype=bool
        )
        new_anchor_array[list(anchor_ids)] = True
        training_ligand["anchors"] = torch.tensor(
            new_anchor_array
        ).float()

        return training_ligand

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
    md_trajectory: dict[str, list[tuple[int, str]]] = field(default_factory=dict, repr=False)
    """Structures of the molecule produced during molecular dynamics simulations.

    The key of the dictionary is the name of the method (e.g., forcefield) used
    for the molecular dynamics.
    The values are a list of pairs of the MD timestep and the structure at that timestep"""

    # Properties
    gas_storage: dict[str, float | tuple[float, float]] = field(default_factory=dict, repr=False)  # TODO (wardlt): Allow only one type of value
    """Storage capacity of the MOF for different gases and pressures. Key is the name of the gas, value is a single capacity value
     or the capacity at different pressures (units TBD)"""
    structure_stability: dict[str, float] = field(default_factory=dict, repr=False)
    """How likely the structure is to be stable according to different assays.
    The value is the strain between the first and last timestep. Larger values indicate worse stability

    The strain for each assay should correspond to the strain from the longest trajectory in :attr:`md_trajectory`
    """

    # Tracking provenance of structure
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
