import sys

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

'''
Given a list of SMILES strings, generate 3D conformers in sdf format using RDKit.
Energy minimizes and filters conformers to meet energy window and rms constraints.

Script modified from: https://github.com/dkoes/rdkit-scripts/blob/master/rdconf.py
'''


# convert smiles to sdf
def getRMS(mol, c1, c2):
    (rms, trans) = AllChem.GetAlignmentTransform(mol, mol, c1, c2)
    return rms


def compute_confs_worker(
        smifile,
        sdffile,
        pid,
        maxconfs=20,
        sample_mult=1,
        seed=42,
        rms_threshold=0.7,
        energy=10,
        verbose=False,
        mmff=False,
        nomin=False,
        etkdg=False,
        smi_frags=[],
        jpsettings=False,
):
    with open(sdffile, 'w+') as outf:
        sdwriter = Chem.SDWriter(outf)
        if sdwriter is None:
            print("Could not open ", sdffile)
            sys.exit(-1)

        if verbose:
            print("Generating a maximum of", maxconfs, "per mol")

        if etkdg and not AllChem.ETKDG:
            print("ETKDB does not appear to be implemented. Please upgrade RDKit.")
            sys.exit(1)

        if smi_frags != []:
            if len(smifile) != len(smi_frags):
                print("smifile and smi_frags not equal in length")
                return None

        # Set clustering and sampling as per https://pubs.acs.org/doi/10.1021/ci2004658
        if jpsettings:
            rms_threshold = 0.35
            sample_mult = 1

        generator = tqdm(enumerate(smifile), total=len(smifile)) if pid == 0 else enumerate(smifile)
        for count, smi in generator:
            name = smi
            pieces = smi.split('.')
            if len(pieces) > 1:
                smi = max(pieces, key=len)  # take largest component by length
                if verbose:
                    print("Taking largest component: %s" % smi)

            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                if verbose:
                    print(smi)
                try:
                    Chem.SanitizeMol(mol)
                    mol = Chem.AddHs(mol)
                    mol.SetProp("_Name", name)
                    if jpsettings:
                        rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
                        if rot_bonds <= 7:
                            maxconfs = 50
                        elif rot_bonds >= 8 and rot_bonds <= 12:
                            maxconfs = 200
                        else:
                            maxconfs = 300
                    if smi_frags != []:
                        mol.SetProp("_StartingPoint", smi_frags[count])

                    if etkdg:
                        cids = AllChem.EmbedMultipleConfs(mol, numConfs=int(sample_mult * maxconfs),
                                                          useExpTorsionAnglePrefs=True, useBasicKnowledge=True,
                                                          randomSeed=seed, numThreads=1)
                    else:
                        cids = AllChem.EmbedMultipleConfs(mol, int(sample_mult * maxconfs), randomSeed=seed, numThreads=1)
                    if verbose:
                        print(len(cids), "conformers found")
                    cenergy = []
                    if mmff:
                        converged_res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
                        cenergy = [i[1] for i in converged_res]
                    elif not nomin and not mmff:
                        converged_res = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=1)
                        cenergy = [i[1] for i in converged_res]
                    else:
                        for conf in cids:
                            # not passing confID only minimizes the first conformer
                            if nomin:
                                cenergy.append(conf)
                            elif mmff:
                                converged = AllChem.MMFFOptimizeMolecule(mol, confId=conf)
                                mp = AllChem.MMFFGetMoleculeProperties(mol)
                                cenergy.append(AllChem.MMFFGetMoleculeForceField(mol, mp, confId=conf).CalcEnergy())
                            else:
                                converged = not AllChem.UFFOptimizeMolecule(mol, confId=conf)
                                cenergy.append(AllChem.UFFGetMoleculeForceField(mol, confId=conf).CalcEnergy())
                            if verbose:
                                print("Convergence of conformer", conf, converged)

                    mol = Chem.RemoveHs(mol)
                    sortedcids = sorted(cids, key=lambda cid: cenergy[cid])
                    if len(sortedcids) > 0:
                        mine = cenergy[sortedcids[0]]
                    else:
                        mine = 0
                    if (rms_threshold == 0):
                        cnt = 0
                        for conf_num, conf in enumerate(sortedcids):
                            if (cnt >= maxconfs):
                                break
                            if (energy < 0) or cenergy[conf] - mine <= energy:
                                mol.SetProp("_Model", str(conf_num))
                                mol.SetProp("_Energy", str(cenergy[conf]))
                                sdwriter.write(mol, conf)
                                cnt += 1
                    else:
                        written = {}
                        for conf_num, conf in enumerate(sortedcids):
                            if len(written) >= maxconfs:
                                break
                            # check rmsd
                            passed = True
                            for seenconf in written:
                                rms = getRMS(mol, seenconf, conf)
                                if (rms < rms_threshold) or (energy > 0 and cenergy[conf] - mine > energy):
                                    passed = False
                                    break
                            if passed:
                                written[conf] = True
                                mol.SetProp("_Model", str(conf_num))
                                mol.SetProp("_Energy", str(cenergy[conf]))
                                sdwriter.write(mol, conf)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as e:
                    print("Exception", e)
            else:
                print("ERROR:", smi)

        sdwriter.close()
