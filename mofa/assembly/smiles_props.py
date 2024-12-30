import json
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen
from rdkit.Chem import GraphDescriptors
from rdkit.Chem import Fragments
from rdkit import DataStructs
from rdkit.DataStructs import BulkTanimotoSimilarity
from mofa.assembly.sascore import SAscore

#SAScore model, taking a SMILES string as input.
SAscore_model = SAscore()

#Load in HMOF fingerprints for Bulk Tanimoto similarity comparison
curr_path = os.path.dirname(os.path.realpath(__file__))
hmof_data_dict = json.load(open(os.path.join(curr_path,'full_hmof_ligand_properties.json'),'r'))
all_hmof_FP = [''.join([str(j) for j in i]) for i in hmof_data_dict['hmof_fps']]
all_hmof_FP = [DataStructs.CreateFromBitString(i) for i in all_hmof_FP]
polaris_data_dict = json.load(open(os.path.join(curr_path,'full_polaris_ligand_properties.json'),'r'))

fpgen = AllChem.GetRDKitFPGenerator()
def mol_to_hmof_tani(mol):
    fp = fpgen.GetFingerprint(mol)
    return max(BulkTanimotoSimilarity(fp,all_hmof_FP))

#RDKit descriptors that comprise ligand embedding
descriptor_funcs = {
        'Molecular_Weight': Descriptors.MolWt,
        'LogP': Descriptors.MolLogP,
        'Num_H_Donors': Descriptors.NumHDonors,
        'Num_H_Acceptors': Descriptors.NumHAcceptors,
        'TPSA': rdMolDescriptors.CalcTPSA,
        'Num_Rotatable_Bonds': Lipinski.NumRotatableBonds,
        'Molar_Refractivity': Crippen.MolMR,
        'Heavy_Atom_Count': Descriptors.HeavyAtomCount,
        'NHOH_Count': Descriptors.NHOHCount,
        'NO_Count': Descriptors.NOCount,
        'Num_Alipathic_Rings': Descriptors.NumAliphaticRings,
        'Num_Aromatic_Rings': Descriptors.NumAromaticRings,
        'Num_Saturated_Rings': Descriptors.NumSaturatedRings,
        'Num_Heteroatoms': Descriptors.NumHeteroatoms,
        'Balaban_J': GraphDescriptors.BalabanJ,
        'LabuteASA': rdMolDescriptors.CalcLabuteASA,
        'Chi0v': GraphDescriptors.Chi0v,
        'Chi1v': GraphDescriptors.Chi1v,
        'HallKierAlpha': Descriptors.HallKierAlpha,
        'Kappa1': Descriptors.Kappa1,
        'Num_Valence_Electrons': Descriptors.NumValenceElectrons,
        'Num_Aromatic_Heterocycles': Descriptors.NumAromaticHeterocycles,
        'Num_Aromatic_Carbocycles': Descriptors.NumAromaticCarbocycles,
        'Num_SpiroAtoms': rdMolDescriptors.CalcNumSpiroAtoms,
        'Num_Bridgehead_Atoms': rdMolDescriptors.CalcNumBridgeheadAtoms,
        'clogp': Crippen.MolLogP,
        'mr': Crippen.MolMR,
        'Chi0n': rdMolDescriptors.CalcChi0n,
        'Chi1n': rdMolDescriptors.CalcChi1n,
        'Chi2n': rdMolDescriptors.CalcChi2n,
        'Chi3n': rdMolDescriptors.CalcChi3n,
        'Chi4n': rdMolDescriptors.CalcChi4n,
        'Chi0v': rdMolDescriptors.CalcChi0v,
        'Chi1v': rdMolDescriptors.CalcChi1v,
        'Chi2v': rdMolDescriptors.CalcChi2v,
        'Chi3v': rdMolDescriptors.CalcChi3v,
        'Chi4v': rdMolDescriptors.CalcChi4v,
        'fracsp3': rdMolDescriptors.CalcFractionCSP3,
        'Hall_Kier_Alpha': rdMolDescriptors.CalcHallKierAlpha,
        'Kappa1': rdMolDescriptors.CalcKappa1,
        'Kappa2': rdMolDescriptors.CalcKappa2,
        'Kappa3': rdMolDescriptors.CalcKappa3,
        'LabuteASA': rdMolDescriptors.CalcLabuteASA,
        'Number_Aliphatic_Rings': rdMolDescriptors.CalcNumAliphaticRings,
        'Number_Aromatic_Rings': rdMolDescriptors.CalcNumAromaticRings,
        'Number_Amide_Bonds': rdMolDescriptors.CalcNumAmideBonds,
        'Number_Atom_Stereocenters': rdMolDescriptors.CalcNumAtomStereoCenters,
        'Number_BridgeHead_Atoms': rdMolDescriptors.CalcNumBridgeheadAtoms,
        'Number_HBA': rdMolDescriptors.CalcNumHBA,
        'Number_HBD': rdMolDescriptors.CalcNumHBD,
        'Number_Hetero_Atoms': rdMolDescriptors.CalcNumHeteroatoms,
        'Number_Hetero_Cycles': rdMolDescriptors.CalcNumHeterocycles,
        'Number_Rings': rdMolDescriptors.CalcNumRings,
        'Number_Rotatable_Bonds':rdMolDescriptors.CalcNumRotatableBonds,
        'Number_Spiro': rdMolDescriptors.CalcNumSpiroAtoms,
        'Number_Saturated_Rings': rdMolDescriptors.CalcNumSaturatedRings,
        'Number_Heavy_Atoms': Lipinski.HeavyAtomCount,
        'Number_NH_OH': Lipinski.NHOHCount,
        'Number_N_O': Lipinski.NOCount,
        'Number_Valence_Electrons': Descriptors.NumValenceElectrons,
        'Max_Partial_Charge': Descriptors.MaxPartialCharge,
        'Min_Partial_Charge': Descriptors.MinPartialCharge,
    }

#Create RDKit embedding from mol object
def mol_to_embed(mol):
    smiles_list = []    
    for i in descriptor_funcs.keys():
        try:
            smiles_list.append(float(descriptor_funcs[i](mol)))
        except ValueError:
            #Sometimes, particularly malformed SMILES, RDKit will throw an error.
            #This has primarily been observed when counting stereocenters, for which
            #it appears most molecules contain none. So if a ValueError occurs,
            #we opt to pad with a zero (for now).
            smiles_list.append(0.0)
    return smiles_list

def init_learning_inputs():
    X = np.concatenate([hmof_data_dict['hmof_rdkit_embed'],polaris_data_dict['polaris_rdkit_embed']])
    y = np.concatenate([hmof_data_dict['hmof_stability'],polaris_data_dict['polaris_stability']])
    X_sub = []
    y_sub = []
    for i in range(len(y)):
        if y[i] is not None:
            X_sub.append(X[i])
            y_sub.append(y[i])
    X_sub = np.array(X_sub)
    y_sub = np.array(y_sub)
    return X_sub,y_sub
