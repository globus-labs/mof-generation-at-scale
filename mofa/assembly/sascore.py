from __future__ import print_function, division
from rdkit import Chem
from rdkit import rdBase
from rdkit.six import iteritems
from rdkit.Chem import rdMolDescriptors

import math
import pickle
import os
rdBase.DisableLog('rdApp.error')

class SAscore():
    """
    Calculation of synthetic accessibility score as described in:
    Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions
    Peter Ertl and Ansgar Schuffenhauer
    Journal of Cheminformatics 1:8 (2009)
    http://www.jcheminf.com/content/1/1/8
    """
    def __init__(self):
        global _fscores
        _fscores = None
    def __call__(self, smile):
        if _fscores is None:
            self.readFragmentScores()
        m = Chem.MolFromSmiles(smile)
        if m:
            try:
                # fragment score
                fp = rdMolDescriptors.GetMorganFingerprint(m, 2)  #<- 2 is the *radius* of the circular fingerprint
                fps = fp.GetNonzeroElements()
                score1 = 0.
                nf = 0
                for bitId, v in iteritems(fps):
                    nf += v
                    sfp = bitId
                    score1 += _fscores.get(sfp, -4)*v
                score1 /= nf
                
                # features score
                nAtoms = m.GetNumAtoms()
                nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
                ri = m.GetRingInfo()
                nBridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(m)
                nSpiro = nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(m)
                nMacrocycles=0
                for x in ri.AtomRings():
                    if len(x)>8: 
                        nMacrocycles+=1
                
                sizePenalty = nAtoms**1.005 - nAtoms
                stereoPenalty = math.log10(nChiralCenters+1)
                spiroPenalty = math.log10(nSpiro+1)
                bridgePenalty = math.log10(nBridgeheads+1)
                macrocyclePenalty = 0.
                # ---------------------------------------
                # This differs from the paper, which defines:
                #  macrocyclePenalty = math.log10(nMacrocycles+1)
                # This form generates better results when 2 or more macrocycles are present
                if nMacrocycles > 0: 
                    macrocyclePenalty = math.log10(2)
                score2 = 0. -sizePenalty -stereoPenalty -spiroPenalty -bridgePenalty -macrocyclePenalty
                # correction for the fingerprint density
                # not in the original publication, added in version 1.1
                # to make highly symmetrical molecules easier to synthetise
                score3 = 0.
                if nAtoms > len(fps):
                    score3 = math.log(float(nAtoms) / len(fps)) * .5
                sascore = score1 + score2 + score3
                
                # need to transform "raw" value into scale between 1 and 10
                min_score = -4.0
                max_score = 2.5
                sascore = 11. - (sascore - min_score + 1) / (max_score - min_score) * 9.
                # smooth the 10-end
                if sascore > 8.: sascore = 8. + math.log(sascore+1.-9.)
                if sascore > 10.: sascore = 10.0
                elif sascore < 1.: sascore = 1.0 
                sascore = math.exp(1 -sascore) # minimize the sascore
                return sascore
            except:
                return 0.0
        else:
             return 0.0   
    
    def readFragmentScores(self, name='fpscores'):
        import gzip
        global _fscores
        # generate the full path filename:
        #name = os.path.join('./', name)
        curr_folder = os.path.split(os.path.abspath(__file__))[0]
        name = os.path.join(curr_folder,name)
        _fscores = pickle.load(gzip.open('%s.pkl.gz'%name))
        outDict = {}
        for i in _fscores:
            for j in range(1,len(i)):
                outDict[i[j]] = float(i[0])
        _fscores = outDict