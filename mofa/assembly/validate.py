"""Validate and standardize a generated molecule"""
from rdkit import Chem

from mofa.model import LigandDescription
from mofa.utils.xyz import xyz_to_mol


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

        # Update the record, add to ligand queue and prepare it for writing to disk
        record['valid'] = True
        valid_ligands.append(ligand)

    ######### MARCUS ADDITIONS/CHANGES #########
    #ASK LOGAN: should this be broken into a different function?
    #db_len = self.collection.count_documents(filter={}) #HOW SHOULD I GET THE DB FROM HERE?
    #if self.use_al:
    #    if db_len>self.mofa_db_length:
    #        self.mofa_db_length = db_len
    #        if db_len // self.al_retrain_interval > self.al_retrain_counter: #Trigger retrain
    #            self.al_retrain_counter = db_len // self.al_retrain_interval
    #            #Use the document length to check if retraining is needed
    #            cursor = (self.collection.find({}))
    #            new_X = []
    #            new_y = []
    #            for record in cursor:
    #                new_X+=[i['rd_embed'] for i in record['ligands'][1:]] #Skipping first ligand (duplicate)
    #                new_y+=[record['structure_stability']['uff'],record['structure_stability']['uff']]
    #            X = np.concatenate([self.X_init,np.array(new_X)])
    #            y = np.concatenate([self.y_init,np.array(new_y)])
    #            new_xgbr = XGBRegressor()
    #            new_xgbr.fit(X,y)
    #            new_model_file = self.out_dir / f'xgbr_al_params_{str(self.al_retrain_counter).zfill(5)}.json'
    #            new_xgbr.save_model(new_model_file)
    #            self.xgbr = new_xgbr
    #
    #    if len(valid_ligands) == 0:
    #        new_stability_predictions = []
    #        new_tani_sims = []
    #        new_sascores = []
    #    else:
    #        embeddings = np.array([lig.rd_embed for lig in valid_ligands])
    #        new_stability_predictions = self.xgbr.predict(embeddings)
    #        new_tani_sims = np.array([lig.hmof_tani_sim for lig in valid_ligands])
    #        new_sascores = np.array([lig.sascore for lig in valid_ligands])
    #
    #    for lig_ind in range(len(valid_ligands)):
    #        valid_ligands[lig_ind].metadata['pred_stability'] = float(new_stability_predictions[lig_ind])
    #
    #    all_lig_list = list(self.ligand_assembly_queue[anchor_type]) + valid_ligands
    #    ordered_inds = acquisition_reorder(all_lig_list,self.al_acquisition) #Reorder according to acquisition function
    #    reordered_ligands_list = [all_lig_list[lig_ind] for lig_ind in ordered_inds]
    #else:
    #    reordered_ligands_list = list(self.ligand_assembly_queue[anchor_type]) + valid_ligands

    # Append the ligands to the task queue
    #self.ligand_assembly_queue[anchor_type].extend(reordered_ligands_list)  # Shoves old ligands out of the deque
    ######### END OF MARCUS ADDITIONS/CHANGES #########

    return valid_ligands, all_records
