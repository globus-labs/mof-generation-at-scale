"""Validate and standardize a generated molecule"""
from rdkit import Chem

from mofa.model import LigandDescription
from mofa.utils.xyz import xyz_to_mol
from mofa.assembly.smiles_props import *



def process_ligands(ligands: list[LigandDescription]) -> tuple[list[LigandDescription], list[dict]]:
    """Assess whether a ligand is valid and prepare it for the next step

    Args:
        ligands: Ligands to be processed
    Returns:
        - List of the ligands which pass validation
        - Records describing the ligands suitable for serialization into CSV file
    """

    all_records = []
    valid_ligands = []

    for ligand in ligands:

        # Store the ligand information for debugging purposes
        record = {"anchor_type": ligand.anchor_type,
                  "name": ligand.name,
                  "smiles": None,
                  "xyz": ligand.xyz,
                  "prompt_atoms": ligand.prompt_atoms,
                  "hmof_tani_sim": None,
                  "sascore": None,
                  "rd_embed": None,
                  "valid": False}
        all_records.append(record)  # Record is still editable even after added to list

        # Try constrained optimization on the ligand
        try:
            ligand.full_ligand_optimization()
        except (ValueError, AttributeError,):
            continue

        # Parse each new ligand, determine whether it is a single molecule
        try:
            mol = xyz_to_mol(ligand.xyz)
        except (ValueError,):
            continue

        # Store the smiles string
        mol = Chem.RemoveHs(mol)
        smiles = Chem.MolToSmiles(mol)
        record['smiles'] = smiles

        if len(Chem.GetMolFrags(mol)) > 1:
            continue

        # If passes, save the SMILES string and store the molecules
        ligand.smiles = Chem.MolToSmiles(mol)

        # Add subsequent properties, assuming everything else has passed
        hmof_tani_sim = mol_to_hmof_tani(Chem.AddHs(mol))
        record['hmof_tani_sim'] = hmof_tani_sim
        ligand.hmof_tani_sim = hmof_tani_sim
        sascore = SAscore_model(ligand.smiles)
        record['sascore'] = sascore
        ligand.sascore = sascore
        rd_embed = mol_to_embed(Chem.AddHs(mol))
        record['rd_embed'] = rd_embed
        ligand.rd_embed = rd_embed

        # Update the record, add to ligand queue and prepare it for writing to disk
        record['valid'] = True
        valid_ligands.append(ligand)
    return valid_ligands, all_records
