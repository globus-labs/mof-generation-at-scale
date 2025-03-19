import numpy as np

def explore_only(ligands):
    stability_ordered_inds = np.argsort([lig.metadata['pred_stability_std'] for lig in ligands])[::-1]
    return stability_ordered_inds

def upper_confidence_bound(ligands,lambda_val=0.1):
    stability_ordered_inds = np.argsort([lig.metadata['pred_stability'] - \
                            lambda_val*lig.metadata['pred_stability_std'] for lig in ligands])[::-1]
    return stability_ordered_inds

def multi_feature(ligands,stability_weight=1.0, tani_weight=0.0, sascore_weight=0.0):
    stability_ordered_inds = np.argsort([stability_weight*lig.metadata['pred_stability'] + \
                    sascore_weight*lig.sascore + \
                    tani_weight*lig.hmof_tani_sim for lig in all_lig_list])[::-1]
    return stability_ordered_inds

def acquisition_reorder(ligands,acq_func):
    allowed_acq_functions = ['explore_only','exploit_only','ucb_lambda01','ucb_lambda20',\
        'tani_only','sascore_only','stability_tani','stability_sascore','stability_tani_sascore']
    if acq_func not in allowed_acq_functions:
        print(f'Warning: Acquisition function {acq_func} not in allowed list. Defaulting to UCB (lambda=0.1).')
        acq_func = 'ucb_lambda01'
    if acq_func == 'explore_only':
        return explore_only(ligands)
    elif acq_func == 'exploit_only':
        return multi_feature(ligands,stability_weight=1.0,tani_weight=0.0,sascore_weight=0.0)
    elif acq_func == 'ucb_lambda01':
        return upper_confidence_bound(ligands,lambda_val=0.1)
    elif acq_func == 'ucb_lambda20':
        return upper_confidence_bound(ligands,lambda_val=2.0)
    elif acq_func == 'tani_only':
        return multi_feature(ligands,stability_weight=0.0,tani_weight=1.0,sascore_weight=0.0)
    elif acq_func == 'sascore_only':
        return multi_feature(ligands,stability_weight=0.0,tani_weight=0.0,sascore_weight=1.0)
    elif acq_func == 'stability_tani':
        return multi_feature(ligands,stability_weight=0.5,tani_weight=0.5,sascore_weight=0.0)
    elif acq_func == 'stability_sascore':
        return multi_feature(ligands,stability_weight=0.5,tani_weight=0.0,sascore_weight=0.5)
    elif acq_func == 'stability_tani_sascore':
        return multi_feature(ligands,stability_weight=1.0/3,tani_weight=1.0/3,sascore_weight=1.0/3)