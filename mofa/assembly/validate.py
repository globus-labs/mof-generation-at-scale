"""Validate and standardize a generated molecule"""
from rdkit import Chem

from mofa.model import LigandDescription
from mofa.utils.xyz import xyz_to_mol
from mofa.utils.difflinker_sample_and_analyze import DiffLinkerOutput


def check_ligand(ligand: LigandDescription) -> tuple[LigandDescription, dict]:
    """Check whether an individual ligand description is satifactory

    Args:
        ligand: Ligand to be evaluated
    Returns:
        - Modified description
        - Record describing it, a dict which contains:
            - anchor_type: The type of the anchor for the ligand
            - name: Name assigned by generator
            - smiles: SMILES string (if detectible)
            - xyz: Geometry
            - prompt_atoms: Indices of the atoms used in the prompt
            - valid: Whether the molecule was valid
    """
    # Store the ligand information for debugging purposes
    record = {"anchor_type": ligand.anchor_type,
              "name": ligand.name,
              "smiles": None,
              "xyz": ligand.xyz,
              "prompt_atoms": ligand.prompt_atoms,
              "valid": False}

    # Try constrained optimization on the ligand
    try:
        ligand.full_ligand_optimization()
    except (ValueError, AttributeError,):
        return ligand, record

    # Parse each new ligand, determine whether it is a single molecule
    try:
        mol = xyz_to_mol(ligand.xyz)
    except (ValueError,):
        return ligand, record

    # Store the smiles string
    mol = Chem.RemoveHs(mol)
    smiles = Chem.MolToSmiles(mol)
    record['smiles'] = smiles

    if len(Chem.GetMolFrags(mol)) > 1:
        return ligand, record

    # If passes, save the SMILES string and store the molecules
    ligand.smiles = Chem.MolToSmiles(mol)

    # Update the record, add to ligand queue and prepare it for writing to disk
    record['valid'] = True
    return ligand, record


def process_ligands(ligands: list[DiffLinkerOutput]) -> tuple[list[LigandDescription], list[dict]]:
    """Assess whether a ligand is valid and prepare it for the next step

    Args:
        ligands: Ligands to be analyzed
    Returns:
        - List of the ligands which pass validation
        - Records describing the ligands suitable for serialization into CSV file
    """

    all_records = []
    valid_ligands = []

    for template, symbols, coords in ligands:
        # Generate the description from the template
        ligand = template.create_description(symbols, coords)
        ligand, record = check_ligand(ligand)
        if record['valid']:
            valid_ligands.append(ligand)
        all_records.append(record)
    return valid_ligands, all_records
