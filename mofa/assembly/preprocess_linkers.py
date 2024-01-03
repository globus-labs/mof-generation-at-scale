"""Convert molecules produced by the generator into forms usable by assembly code"""

import itertools
from rdkit import Chem
from rdkit.Chem.rdchem import RWMol, Mol
from rdkit.Chem.rdmolops import GetMolFrags, GetShortestPath, RemoveHs, AddHs, SanitizeMol
from rdkit.Chem.AllChem import EmbedMolecule, UFFOptimizeMolecule

NUM2MAXNEIGH = {
    "X": 0,
    "C": 3,
    "N": 3,
    "O": 2,
    "F": 1,
    "Cl": 1,
    "Br": 1,
    "At": 1,
    "Fr": 1}


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
        atom.SetProp("atomLabel", atom.GetSymbol() + ":" + str(atom.GetIdx()))
    return mol


def ring2bonds(ring):
    return set([tuple(sorted((ring[i], ring[(i + 1) % len(ring)])))
                for i in range(0, len(ring))])


def bulkRemoveAtoms(emol, atoms2rm):
    emol_copy = RWMol(emol)
    for a2rm in atoms2rm:
        emol_copy.GetAtomWithIdx(a2rm).SetAtomicNum(0)
    emol_copy = Chem.DeleteSubstructs(emol_copy, Chem.MolFromSmarts('[#0]'))
    emol = RWMol(emol_copy)
    return emol


def bulkRemoveBonds(emol, bonds2rm, fragAllowed=False):
    emol_copy = RWMol(emol)
    for b2rm in bonds2rm:
        _emol_copy = RWMol(emol_copy)
        _emol_copy.RemoveBond(b2rm[0], b2rm[1])
        if not fragAllowed:
            if len(GetMolFrags(_emol_copy)) == 1:
                emol_copy = RWMol(_emol_copy)
        else:
            emol_copy = RWMol(_emol_copy)
    emol = RWMol(emol_copy)
    return emol


def rdkitGetLargestCC(emol):
    emol = emol.GetMol()
    atoms2rm = list(
        itertools.chain(
            *
            sorted(
                GetMolFrags(emol),
                key=lambda x: len(x),
                reverse=True)[1:])
    )
    emol = RWMol(emol)
    emol = bulkRemoveAtoms(emol, atoms2rm)
    return emol


def getLongestPath(emol):
    maxPath = []
    for x1 in emol.GetAtoms():
        if x1.GetSymbol() == "C":
            for x2 in emol.GetAtoms():
                if x2.GetSymbol() == "C":
                    if x1.GetIdx() < x2.GetIdx():
                        p = GetShortestPath(emol, x1.GetIdx(), x2.GetIdx())
                        if len(p) > len(maxPath):
                            maxPath = p[:]
    return maxPath


def realSanitizeMol(emol):
    ssr = Chem.GetSymmSSSR(emol)
    urs = [ring2bonds(list(x)) for x in ssr if len(list(x)) == 6]
    if len(urs) > 0:
        ring_bonds = set.union(*urs)
    else:
        ring_bonds = set()
    blist = [tuple(sorted([x.GetEndAtomIdx(), x.GetBeginAtomIdx()]))
             for x in emol.GetBonds()]
    for b in blist:
        if b in list(ring_bonds):
            emol.RemoveBond(b[0], b[1])
            emol.AddBond(b[0], b[1], Chem.rdchem.BondType.AROMATIC)
    Chem.SanitizeMol(emol)
    return emol


def makeRigid(emol):
    ssr = Chem.GetSymmSSSR(emol)
    urs = [ring2bonds(list(x)) for x in ssr if len(list(x)) == 6]
    if len(urs) > 0:
        ring_bonds = set.union(*urs)
    else:
        ring_bonds = set()
    blist = [tuple(sorted([x.GetEndAtomIdx(), x.GetBeginAtomIdx()]))
             for x in emol.GetBonds()]
    for _b in blist:
        if _b not in ring_bonds:
            b = emol.GetBondBetweenAtoms(_b[0], _b[1])
            b1 = b.GetEndAtom()
            b2 = b.GetBeginAtom()
            if not isinstance(
                    emol.GetBondBetweenAtoms(
                        _b[0], _b[1]), type(None)):
                if b1.GetSymbol() == "C":
                    b1maxneigh = 4
                else:
                    b1maxneigh = NUM2MAXNEIGH[b1.GetSymbol()]
                if b2.GetSymbol() == "C":
                    b2maxneigh = 4
                else:
                    b2maxneigh = NUM2MAXNEIGH[b2.GetSymbol()]
                if b1.GetDegree() - b1maxneigh <= -2 and b2.GetDegree() - b2maxneigh <= -2:
                    if emol.GetBondBetweenAtoms(
                            _b[0], _b[1]).GetBondTypeAsDouble() != 3.:
                        emol.RemoveBond(_b[0], _b[1])
                        emol.AddBond(_b[0], _b[1], Chem.rdchem.BondType.TRIPLE)
                elif b1.GetDegree() - NUM2MAXNEIGH[b1.GetSymbol()] <= -1 and b2.GetDegree() - NUM2MAXNEIGH[b1.GetSymbol()] <= -1:
                    if emol.GetBondBetweenAtoms(
                            _b[0], _b[1]).GetBondTypeAsDouble() != 2.:
                        emol.RemoveBond(_b[0], _b[1])
                        emol.AddBond(_b[0], _b[1], Chem.rdchem.BondType.DOUBLE)
    return emol


def convert2COOLinker(emol, dummyElement="At"):
    dummyAtomicNum = Chem.rdchem.Atom(dummyElement).GetAtomicNum()
    emol = Chem.MolFromSmiles(Chem.MolToSmiles(emol))
    emol = AddHs(Chem.Mol(emol), addCoords=True)
    EmbedMolecule(emol)
    UFFOptimizeMolecule(emol)
    conf = emol.GetConformer()

    emol = RemoveHs(Chem.Mol(emol))
    sp2spC = [x.GetIdx() for x in emol.GetAtoms() if
              x.GetSymbol() == "C" and
              (x.GetHybridization() == Chem.rdchem.HybridizationType.SP2 or
               x.GetHybridization() == Chem.rdchem.HybridizationType.SP)]
    sp2spCUnSat = [x.GetIdx() for x in emol.GetAtoms() if
                   x.GetSymbol() == "C" and
                   (x.GetHybridization() == Chem.rdchem.HybridizationType.SP2 or
                    x.GetHybridization() == Chem.rdchem.HybridizationType.SP) and
                   x.GetDegree() < NUM2MAXNEIGH[x.GetSymbol()]]
    maxDist = 0
    maxPair = (0, 0)
    for a1_i in range(0, len(sp2spCUnSat) - 1):
        for a2_i in range(a1_i + 1, len(sp2spCUnSat)):
            a1 = sp2spCUnSat[a1_i]
            a2 = sp2spCUnSat[a2_i]
            dist = conf.GetAtomPosition(a1).Distance(conf.GetAtomPosition(a2))
            if dist > maxDist:
                if set(GetShortestPath(emol, a1, a2)).issubset(set(sp2spC)):
                    maxDist = dist
                    maxPair = (a1, a2)
    # emol = AddHs(Chem.Mol(emol), addCoords=True)
    SanitizeMol(emol)
    emol = AddHs(Chem.Mol(emol), addCoords=True)
    emol = RWMol(emol)
    [x for x in emol.GetAtomWithIdx(maxPair[0]).GetNeighbors(
    ) if x.GetSymbol() == "H"][0].SetAtomicNum(dummyAtomicNum)
    [x for x in emol.GetAtomWithIdx(maxPair[1]).GetNeighbors(
    ) if x.GetSymbol() == "H"][0].SetAtomicNum(dummyAtomicNum)
    mod_mol = Chem.ReplaceSubstructs(
        emol,
        Chem.MolFromSmiles(
            'C[' + dummyElement + ']'),
        Chem.MolFromSmiles('CC(=O)O'),
        replaceAll=True)[0]
    # mod_mol = Chem.MolFromSmiles(Chem.MolToSmiles(emol).replace('[' + dummyElement + ']', 'C(=S)S'))
    emol = RWMol(mod_mol)
    emol = AddHs(Chem.Mol(mod_mol), addCoords=True)
    SanitizeMol(emol)
    EmbedMolecule(emol)
    UFFOptimizeMolecule(emol)
    conf = emol.GetConformer()

    match_ids = emol.GetSubstructMatch(Chem.MolFromSmiles('C(=O)O'))
    carboxylicC = [at for at in match_ids if emol.GetAtomWithIdx(
        at).GetSymbol() == "C"][0]
    bonds2rm = [tuple(sorted((at, carboxylicC)))
                for at in match_ids if emol.GetAtomWithIdx(at).GetSymbol() == "O"]
    emol.GetAtomWithIdx(carboxylicC).SetAtomicNum(dummyAtomicNum)
    emol = bulkRemoveBonds(emol, bonds2rm, fragAllowed=True)
    emol = rdkitGetLargestCC(emol)
    SanitizeMol(emol)
    match_ids = emol.GetSubstructMatch(Chem.MolFromSmiles('C(=O)O'))
    carboxylicC = [at for at in match_ids if emol.GetAtomWithIdx(
        at).GetSymbol() == "C"][0]
    bonds2rm = [tuple(sorted((at, carboxylicC)))
                for at in match_ids if emol.GetAtomWithIdx(at).GetSymbol() == "O"]
    emol.GetAtomWithIdx(carboxylicC).SetAtomicNum(dummyAtomicNum)
    emol = bulkRemoveBonds(emol, bonds2rm, fragAllowed=True)
    emol = rdkitGetLargestCC(emol)
    SanitizeMol(emol)
    return emol


def convert2PyridineLinker(emol, dummyElement="Fr"):
    dummyAtomicNum = Chem.rdchem.Atom(dummyElement).GetAtomicNum()
    emol = Chem.MolFromSmiles(Chem.MolToSmiles(emol))
    emol = AddHs(Chem.Mol(emol), addCoords=True)
    EmbedMolecule(emol)
    UFFOptimizeMolecule(emol)
    conf = emol.GetConformer()

    emol = RemoveHs(Chem.Mol(emol))
    sp2spC = [x.GetIdx() for x in emol.GetAtoms() if
              x.GetSymbol() == "C" and
              (x.GetHybridization() == Chem.rdchem.HybridizationType.SP2 or
               x.GetHybridization() == Chem.rdchem.HybridizationType.SP)]
    sp2spCUnSat = [x.GetIdx() for x in emol.GetAtoms() if
                   x.GetSymbol() == "C" and
                   (x.GetHybridization() == Chem.rdchem.HybridizationType.SP2 or
                    x.GetHybridization() == Chem.rdchem.HybridizationType.SP) and
                   x.GetDegree() < NUM2MAXNEIGH[x.GetSymbol()]]
    maxDist = 0
    maxPair = (0, 0)
    for a1_i in range(0, len(sp2spCUnSat) - 1):
        for a2_i in range(a1_i + 1, len(sp2spCUnSat)):
            a1 = sp2spCUnSat[a1_i]
            a2 = sp2spCUnSat[a2_i]
            dist = conf.GetAtomPosition(a1).Distance(conf.GetAtomPosition(a2))
            if dist > maxDist:
                if set(GetShortestPath(emol, a1, a2)).issubset(set(sp2spC)):
                    maxDist = dist
                    maxPair = (a1, a2)
    # emol = AddHs(Chem.Mol(emol), addCoords=True)
    SanitizeMol(emol)
    emol = AddHs(Chem.Mol(emol), addCoords=True)
    emol = RWMol(emol)
    [x for x in emol.GetAtomWithIdx(maxPair[0]).GetNeighbors(
    ) if x.GetSymbol() == "H"][0].SetAtomicNum(dummyAtomicNum)
    [x for x in emol.GetAtomWithIdx(maxPair[1]).GetNeighbors(
    ) if x.GetSymbol() == "H"][0].SetAtomicNum(dummyAtomicNum)
    mod_mol = Chem.ReplaceSubstructs(
        emol,
        Chem.MolFromSmiles(
            'C[' + dummyElement + ']'),
        Chem.MolFromSmiles('Cc1ccc(S)cc1'),
        replaceAll=True)[0]
    # mod_mol = Chem.MolFromSmiles(Chem.MolToSmiles(emol).replace('[' + dummyElement + ']', 'c1ccc(S)cc1'))
    emol = RWMol(mod_mol)
    emol = AddHs(Chem.Mol(mod_mol), addCoords=True)
    SanitizeMol(emol)
    EmbedMolecule(emol)
    UFFOptimizeMolecule(emol)
    conf = emol.GetConformer()

    extraHs = []

    match_ids = emol.GetSubstructMatch(Chem.MolFromSmiles('c1ccc(S)cc1'))
    paddleCore = [at for at in match_ids if emol.GetAtomWithIdx(
        at).GetSymbol() == "S"][0]
    emol.GetAtomWithIdx(paddleCore).SetAtomicNum(dummyAtomicNum)
    pyridineN = list(emol.GetAtomWithIdx(paddleCore).GetNeighbors())[0]
    pyridineN.SetAtomicNum(7)
    extraHs.append([x for x in emol.GetAtomWithIdx(
        paddleCore).GetNeighbors() if x.GetSymbol() == "H"][0].GetIdx())
    match_ids = emol.GetSubstructMatch(Chem.MolFromSmiles('c1ccc(S)cc1'))
    paddleCore = [at for at in match_ids if emol.GetAtomWithIdx(
        at).GetSymbol() == "S"][0]
    emol.GetAtomWithIdx(paddleCore).SetAtomicNum(dummyAtomicNum)
    pyridineN = list(emol.GetAtomWithIdx(paddleCore).GetNeighbors())[0]
    pyridineN.SetAtomicNum(7)
    extraHs.append([x for x in emol.GetAtomWithIdx(
        paddleCore).GetNeighbors() if x.GetSymbol() == "H"][0].GetIdx())
    emol = bulkRemoveAtoms(emol, extraHs)
    conf = emol.GetConformer()
    return emol


def convert2CyanoLinker(emol, dummyElement="Fr"):
    dummyElement = "Fr"
    dummyAtomicNum = Chem.rdchem.Atom(dummyElement).GetAtomicNum()
    emol = Chem.MolFromSmiles(Chem.MolToSmiles(emol))
    emol = AddHs(Chem.Mol(emol), addCoords=True)
    EmbedMolecule(emol)
    UFFOptimizeMolecule(emol)
    conf = emol.GetConformer()

    emol = RemoveHs(Chem.Mol(emol))
    sp2spC = [x.GetIdx() for x in emol.GetAtoms() if
              x.GetSymbol() == "C" and
              (x.GetHybridization() == Chem.rdchem.HybridizationType.SP2 or
               x.GetHybridization() == Chem.rdchem.HybridizationType.SP)]
    sp2spCUnSat = [x.GetIdx() for x in emol.GetAtoms() if
                   x.GetSymbol() == "C" and
                   (x.GetHybridization() == Chem.rdchem.HybridizationType.SP2 or
                    x.GetHybridization() == Chem.rdchem.HybridizationType.SP) and
                   x.GetDegree() < NUM2MAXNEIGH[x.GetSymbol()]]
    maxDist = 0
    maxPair = (0, 0)
    for a1_i in range(0, len(sp2spCUnSat) - 1):
        for a2_i in range(a1_i + 1, len(sp2spCUnSat)):
            a1 = sp2spCUnSat[a1_i]
            a2 = sp2spCUnSat[a2_i]
            dist = conf.GetAtomPosition(a1).Distance(conf.GetAtomPosition(a2))
            if dist > maxDist:
                if set(GetShortestPath(emol, a1, a2)).issubset(set(sp2spC)):
                    maxDist = dist
                    maxPair = (a1, a2)

    SanitizeMol(emol)
    emol = AddHs(Chem.Mol(emol), addCoords=True)
    emol = RWMol(emol)
    [x for x in emol.GetAtomWithIdx(maxPair[0]).GetNeighbors(
    ) if x.GetSymbol() == "H"][0].SetAtomicNum(dummyAtomicNum)
    [x for x in emol.GetAtomWithIdx(maxPair[1]).GetNeighbors(
    ) if x.GetSymbol() == "H"][0].SetAtomicNum(dummyAtomicNum)
    # display(Chem.MolToSmiles(emol))
    mod_mol = Chem.MolFromSmiles(
        Chem.MolToSmiles(emol).replace(
            '[' + dummyElement + ']', 'C#CS'))
    emol = RWMol(mod_mol)
    # display(Chem.MolToSmiles(mod_mol))
    emol = AddHs(Chem.Mol(mod_mol), addCoords=True)
    SanitizeMol(emol)
    EmbedMolecule(emol)

    UFFOptimizeMolecule(emol)
    conf = emol.GetConformer()
    extraHs = []
    match_ids = emol.GetSubstructMatch(Chem.MolFromSmiles('C#CS'))
    paddleCore = [at for at in match_ids if emol.GetAtomWithIdx(
        at).GetSymbol() == "S"][0]
    emol.GetAtomWithIdx(paddleCore).SetAtomicNum(dummyAtomicNum)
    extraHs.append([x for x in emol.GetAtomWithIdx(
        paddleCore).GetNeighbors() if x.GetSymbol() == "H"][0].GetIdx())
    emol = bulkRemoveAtoms(emol, extraHs)
    cyanoN = list(emol.GetAtomWithIdx(paddleCore).GetNeighbors())[0]
    cyanoN.SetAtomicNum(7)
    conf = emol.GetConformer()
    extraHs = []
    match_ids = emol.GetSubstructMatch(Chem.MolFromSmiles('C#CS'))
    paddleCore = [at for at in match_ids if emol.GetAtomWithIdx(
        at).GetSymbol() == "S"][0]
    emol.GetAtomWithIdx(paddleCore).SetAtomicNum(dummyAtomicNum)
    extraHs.append([x for x in emol.GetAtomWithIdx(
        paddleCore).GetNeighbors() if x.GetSymbol() == "H"][0].GetIdx())
    emol = bulkRemoveAtoms(emol, extraHs)
    cyanoN = list(emol.GetAtomWithIdx(paddleCore).GetNeighbors())[0]
    cyanoN.SetAtomicNum(7)
    conf = emol.GetConformer()
    return emol


def clean_linker(RDKitMOL: Mol) -> dict[str, str]:
    """attach anchors to a molecule such that it becomes a ligand

    Args:
        RDKitMOL: a rdkit.Chem.rdchem.Mol instance of the input molecule

    Returns:
        A dictionary mapping the type of linker to a version of molecule ready for use in that type
    """

    # Prepare the molecule
    emol = Chem.rdchem.EditableMol(RDKitMOL)
    emol = rdkitGetLargestCC(emol)
    emol = realSanitizeMol(emol)
    emol = makeRigid(emol)
    emol = Chem.MolFromSmiles(Chem.MolToSmiles(emol))
    if emol is None:
        raise ValueError(f'Failed to clean linker')
    emol = AddHs(Chem.Mol(emol), addCoords=True)

    # Determine an geometry
    EmbedMolecule(emol)
    UFFOptimizeMolecule(emol)
    emol.GetConformer()

    # Prepare the molecule for use in several applications
    output = {}
    for linker_type, func in [
        ('COO', convert2COOLinker),
        ('pyridine', convert2PyridineLinker),
        ('cyano', convert2CyanoLinker)
    ]:
        try:
            new_emol = func(emol)
        except (ValueError, RuntimeError):
            continue
        output[linker_type] = Chem.MolToXYZBlock(new_emol)

    return output
