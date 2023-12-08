"""Functions for assembling a MOF structure"""
from typing import Sequence
from .model import NodeDescription, LigandDescription, MOFRecord
import os
import io
import itertools
import pandas as pd
import numpy as np
import pymatgen.core as mg
from rdkit import Chem


def readPillaredPaddleWheelXYZ(fpath, dummyElementCOO="At", dummyElementPillar="Fr"):
    """Read xyz file of a paddlewheel node

    Args:
        fpath: xyz file path
        dummyElementCOO: dummy element for anchoring the -C(=O)O group
        dummyElementPillar: dummy element for anchoring the aromatic N group

    Returns:
        Pandas of node [element, x, y, z], -COO anchor ID pairs in the opposite directions, -N anchor ID pairs in the opposite directions
    """
    df = pd.read_csv(fpath, sep=r"\s+", skiprows=2, names=["el", "x", "y", "z"])
    COOAnchorIds = df[df["el"] == dummyElementCOO].index.tolist()
    PillarAnchorIds = df[df["el"] == dummyElementPillar].index.tolist()
    df["maxDist"] = 0.
    anchorPairs = []
    while len(COOAnchorIds) > 0:
        currAnchor = COOAnchorIds[-1]
        COOAnchorIds = COOAnchorIds[:-1]
        df.loc[COOAnchorIds, "maxDist"] = np.linalg.norm(df.loc[currAnchor, ["x", "y", "z"]].astype(float).values -
                                                         df.loc[COOAnchorIds, ["x", "y", "z"]].values, axis=1)

        pairAnchor = df[df["maxDist"] == df.loc[COOAnchorIds, "maxDist"].max()].index[0]
        COOAnchorIds = COOAnchorIds[:COOAnchorIds.index(pairAnchor)] + COOAnchorIds[COOAnchorIds.index(pairAnchor) + 1:]
        anchorPairs.append([currAnchor, pairAnchor])
    return df[["el", "x", "y", "z"]], anchorPairs, PillarAnchorIds


def readTetramerXYZ(fpath, dummyElement="At"):
    """Read xyz file of a tetramer node

    Args:
        fpath: xyz file path
        dummyElementCOO: dummy element for anchoring the -C(=O)O group

    Returns:
        Pandas of node element, x, y, z, and anchor ID pairs
    """
    df = pd.read_csv(fpath, sep=r"\s+", skiprows=2, names=["el", "x", "y", "z"])
    anchorIds = df[df["el"] == dummyElement].index.tolist()
    df["maxDist"] = 0.
    anchorPairs = []
    while len(anchorIds) > 0:
        currAnchor = anchorIds[-1]
        anchorIds = anchorIds[:-1]
        df.loc[anchorIds, "maxDist"] = np.linalg.norm(df.loc[currAnchor, ["x", "y", "z"]].astype(float).values -
                                                      df.loc[anchorIds, ["x", "y", "z"]].values, axis=1)

        pairAnchor = df[df["maxDist"] == df.loc[anchorIds, "maxDist"].max()].index[0]
        anchorIds = anchorIds[:anchorIds.index(pairAnchor)] + anchorIds[anchorIds.index(pairAnchor) + 1:]
        anchorPairs.append([currAnchor, pairAnchor])

    return df[["el", "x", "y", "z"]], anchorPairs


def readCOOLinkerXYZ(fpath, dummyElement="At"):
    """Read xyz file of a -COO type connect-2 ligand

    Args:
        fpath: xyz file path
        dummyElement: dummy element for anchoring the -C(=O)O group

    Returns:
        Pandas of node element, x, y, z, and anchor ID pairs
    """
    df = pd.read_csv(fpath, sep=r"\s+", skiprows=2, names=["el", "x", "y", "z"])
    anchorIds = df[df["el"] == dummyElement].index.tolist()
    return df, anchorIds


def readNbasedLinkerXYZ(fpath, dummyElement="Fr"):
    """Read xyz file of a -N type connect-2 ligand

    Args:
        fpath: xyz file path
        dummyElement: dummy element for anchoring the -N group

    Returns:
        Pandas of node element, x, y, z, and anchor ID pairs
    """
    df = pd.read_csv(fpath, sep=r"\s+", skiprows=2, names=["el", "x", "y", "z"])
    anchorIds = df[df["el"] == dummyElement].index.tolist()
    return df, anchorIds


def pandas2xyzfile(df, fpath, useStringIO=False):
    """write a Pandas with [element, x, y, z] to a xyz file

    Args:
        df: Pandas with [element, x, y, z]
        fpath: xyz file path
        useStringIO: whether to write Pandas content into a string

    Returns:
        string of xyz file if useStringIO enabled, or empty string if useStringIO disabled
    """
    if useStringIO:
        return str(len(df)) + "\n\n" + df.to_string(header=None, index=None)
    else:
        with io.open(fpath, "w", newline="\n") as wf:
            wf.write(str(len(df)) + "\n\n" + df.to_string(header=None, index=None))
        return ""


def rotmat2align(vec1, vec2):
    """return a rotation matrix that rotates a 3D vector to align with another 3D vector

    Args:
        vec1: input vector 1
        vec2: input vector 2

    Returns:
        3x3 rotation matrix
    """
    # https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def assemble_COO_pcuMOF_multiProc(inputDict):
    """multiprocessing wrapper for assemble_COO_pcuMOF

    Args:
        inputDict: input of many structures to assemble in the format of dictionary

    Returns:
        returnValue a file path if assembly succeeded, None if failed
    """
    nodePath = inputDict["nodePath"]
    linkerPaths = inputDict["linkerPaths"]
    newMOFdir = inputDict["newMOFdir"]
    if "dummyElement" in inputDict.keys():
        dummyElement = inputDict["dummyElement"]
    else:
        dummyElement = "At"

    returnValue = None
    try:
        returnValue = assemble_COO_pcuMOF(nodePath, linkerPaths, newMOFdir, dummyElement=dummyElement)
    except Exception as e:
        print(e)
    return returnValue


def assemble_COO_pcuMOF(nodePath, linkerPaths, newMOFpath, dummyElement="At"):
    """assembly code for -COO only MOF with pcu topology

    Args:
        nodePath: node xyz file path,
        linkerPaths: node xyz file paths,
        newMOFpath: assembled MOF file path,
        dummyElement: dummy element for anchoring positions

    Returns:
        returnValue a cif file path if assembly succeeded, None if failed
    """
    MOFRootdir = os.path.split(newMOFpath)[0]
    os.makedirs(MOFRootdir, exist_ok=True)
    Chem.rdchem.Atom(dummyElement).GetAtomicNum()

    node_df, nodeAnchors = readTetramerXYZ(nodePath)
    linkerAnchors = [None for x in range(3)]
    linker_dfs = [None for x in range(3)]
    linker_dfs[0], linkerAnchors[0] = readCOOLinkerXYZ(linkerPaths[0])
    linker_dfs[1], linkerAnchors[1] = readCOOLinkerXYZ(linkerPaths[1])
    linker_dfs[2], linkerAnchors[2] = readCOOLinkerXYZ(linkerPaths[2])
    ldf = []
    latVec = []
    for _i_ in range(len(nodeAnchors)):
        noderAnchorPair = nodeAnchors[_i_]
        nodeAnchor1POS = node_df.loc[noderAnchorPair[0], ["x", "y", "z"]].astype(float).values
        nodePaired2POS = node_df.loc[noderAnchorPair[1], ["x", "y", "z"]].astype(float).values
        anchorPairVec = nodeAnchor1POS - nodePaired2POS
        linker_df = linker_dfs[_i_]
        linkerAnchorPair = linkerAnchors[_i_]

        # translate to align n1 and l1
        linkerAnchor1POS = linker_df.loc[linkerAnchorPair[0], ["x", "y", "z"]].astype(float).values
        linker_df["dist2"] = np.sum((linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor1POS) *
                                    (linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor1POS),
                                    axis=1)
        linkerCarbon1Idx = linker_df[linker_df["dist2"] == linker_df[linker_df["el"] == "C"]["dist2"].min()].index.tolist()[0]

        linkerAnchor2POS = linker_df.loc[linkerAnchorPair[1], ["x", "y", "z"]].astype(float).values
        linker_df["dist2"] = np.sum((linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor2POS) *
                                    (linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor2POS),
                                    axis=1)
        linkerCarbon2Idx = linker_df[linker_df["dist2"] == linker_df[linker_df["el"] == "C"]["dist2"].min()].index.tolist()[0]

        linkerAnchorVec = linkerAnchor1POS - linkerAnchor2POS

        rotmat = rotmat2align(-linkerAnchorVec, anchorPairVec)
        linker_df.loc[:, ["x", "y", "z"]] = rotmat.dot(linker_df.loc[:, ["x", "y", "z"]].values.T).T

        displacementVec = node_df.loc[noderAnchorPair[0], ["x", "y", "z"]].astype(float).values - \
            linker_df.loc[linkerCarbon1Idx, ["x", "y", "z"]].astype(float).values
        linker_df.loc[:, ["x", "y", "z"]] = linker_df.loc[:, ["x", "y", "z"]].values + displacementVec

        ldf.append(linker_df[linker_df["el"] != dummyElement].copy(deep=True))
        farAnchorNLPairDispVec = node_df.loc[noderAnchorPair[1], ["x", "y", "z"]].astype(float).values - \
            linker_df.loc[linkerCarbon2Idx, ["x", "y", "z"]].astype(float).values
        latVec.append(farAnchorNLPairDispVec)
        # pandas2xyzfile(linker_df, os.path.join(newMOFdir, "linker-"+str(_i_)+".xyz"))

    # pandas2xyzfile(node_df, os.path.join(newMOFpath, "node.xyz"))
    final_df = pd.concat(ldf + [node_df[node_df["el"] != dummyElement].copy(deep=True)], axis=0)
    final_df = final_df[final_df["el"] != dummyElement].reset_index(drop=True)[["el", "x", "y", "z"]]
    xyzstr = pandas2xyzfile(final_df, os.path.join(MOFRootdir, "mofCart.xyz"), useStringIO=True)

    mol = mg.Molecule.from_str(xyzstr, fmt="xyz")
    # try:
    #     os.remove(os.path.join(MOFRootdir, "mofCart.xyz"))
    # except:
    #     print(os.path.join(MOFRootdir, "mofCart.xyz") + " does not exist!\n")
    MOFstruct = mg.Structure(np.array(latVec),
                             mol.species,
                             mol.cart_coords,
                             coords_are_cartesian=True)

    df = pd.read_csv("../xyan11-code/OChemDB_bond_threshold.csv", index_col=0)
    element2bondLengthMap = dict(zip(df["element"], df["min"] - (df["stddev"] * 0.01)))
    unique_bond_el = list(set(list(itertools.chain(*[["-".join(sorted([x.symbol, y.symbol])) for x in MOFstruct.species] for y in MOFstruct.species]))))
    unique_bond_el = unique_bond_el + ["Fr-Se"]
    for x in unique_bond_el:
        if x not in element2bondLengthMap:
            element2bondLengthMap[x] = 0.
            for y in x.split("-"):
                if not isinstance(mg.periodic_table.Element(y).atomic_radius_calculated, type(None)):
                    element2bondLengthMap[x] = element2bondLengthMap[x] + mg.periodic_table.Element(y).atomic_radius_calculated
    distMat = MOFstruct.distance_matrix
    distMat[distMat == 0] = np.inf
    bondLengthThresMat = np.array([[element2bondLengthMap["-".join(sorted([x.symbol, y.symbol]))] for x in MOFstruct.species] for y in MOFstruct.species])
    if np.all(distMat > bondLengthThresMat):
        MOFstruct.to(filename=newMOFpath + ".cif", fmt="cif")
        return newMOFpath + ".cif"
    else:
        return ""


def assemble_pillaredPaddleWheel_pcuMOF(nodePath,
                                        COOLinkerPaths,
                                        PillarLinkerPath,
                                        newMOFpath,
                                        dummyElementCOO="At",
                                        dummyElementPillar="Fr"):
    """assembly code for -COO and -N ligands MOF with pcu topology

    Args:
        nodePath: node xyz file path,
        COOLinkerPaths: -COO node xyz file paths,
        PillarLinkerPath: -N node xyz file paths,
        newMOFpath: assembled MOF file path,
        dummyElementCOO: dummy element for -COO anchoring positions,
        dummyElementPillar: dummy element for -N anchoring positions

    Returns:
        returnValue a cif file path if assembly succeeded, None if failed
    """
    MOFRootdir = os.path.split(newMOFpath)[0]
    os.makedirs(MOFRootdir, exist_ok=True)
    Chem.rdchem.Atom(dummyElementCOO).GetAtomicNum()
    Chem.rdchem.Atom(dummyElementPillar).GetAtomicNum()

    node_df, nodeAnchorsCOO, nodeAnchorsPillar = readPillaredPaddleWheelXYZ(nodePath)

    linkerAnchors = [None for x in range(2)]
    linker_dfs = [None for x in range(2)]
    linker_dfs[0], linkerAnchors[0] = readCOOLinkerXYZ(COOLinkerPaths[0])
    linker_dfs[1], linkerAnchors[1] = readCOOLinkerXYZ(COOLinkerPaths[1])
    Pillarlinker_df, PillarlinkerAnchors = readNbasedLinkerXYZ(PillarLinkerPath)
    ldf = []
    latVec = []

    noderAnchorPair = nodeAnchorsPillar
    nodeAnchor1POS = node_df.loc[noderAnchorPair[0], ["x", "y", "z"]].astype(float).values
    nodePaired2POS = node_df.loc[noderAnchorPair[1], ["x", "y", "z"]].astype(float).values
    anchorPairVec = nodeAnchor1POS - nodePaired2POS
    linker_df = Pillarlinker_df
    linkerAnchorPair = PillarlinkerAnchors

    # translate to align n1 and l1
    linkerAnchor1POS = linker_df.loc[linkerAnchorPair[0], ["x", "y", "z"]].astype(float).values
    linker_df["dist2"] = np.sum((linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor1POS) *
                                (linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor1POS),
                                axis=1)
    linkerNitrogen1Idx = linker_df[linker_df["dist2"] == linker_df[linker_df["el"] == "N"]["dist2"].min()].index.tolist()[0]

    linkerAnchor2POS = linker_df.loc[linkerAnchorPair[1], ["x", "y", "z"]].astype(float).values
    linker_df["dist2"] = np.sum((linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor2POS) *
                                (linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor2POS),
                                axis=1)
    linkerNitrogen2Idx = linker_df[linker_df["dist2"] == linker_df[linker_df["el"] == "N"]["dist2"].min()].index.tolist()[0]

    linkerAnchorVec = linkerAnchor1POS - linkerAnchor2POS

    rotmat = rotmat2align(-linkerAnchorVec, anchorPairVec)
    linker_df.loc[:, ["x", "y", "z"]] = rotmat.dot(linker_df.loc[:, ["x", "y", "z"]].values.T).T

    displacementVec = node_df.loc[noderAnchorPair[0], ["x", "y", "z"]].astype(float).values - \
        linker_df.loc[linkerNitrogen1Idx, ["x", "y", "z"]].astype(float).values
    linker_df.loc[:, ["x", "y", "z"]] = linker_df.loc[:, ["x", "y", "z"]].values + displacementVec

    ldf.append(linker_df[linker_df["el"] != dummyElementPillar].copy(deep=True))
    farAnchorNLPairDispVec = node_df.loc[noderAnchorPair[1], ["x", "y", "z"]].astype(float).values - \
        linker_df.loc[linkerNitrogen2Idx, ["x", "y", "z"]].astype(float).values
    latVec.append(farAnchorNLPairDispVec)

    for _i_ in range(len(nodeAnchorsCOO)):
        noderAnchorPair = nodeAnchorsCOO[_i_]
        nodeAnchor1POS = node_df.loc[noderAnchorPair[0], ["x", "y", "z"]].astype(float).values
        nodePaired2POS = node_df.loc[noderAnchorPair[1], ["x", "y", "z"]].astype(float).values
        anchorPairVec = nodeAnchor1POS - nodePaired2POS
        linker_df = linker_dfs[_i_]
        linkerAnchorPair = linkerAnchors[_i_]

        # translate to align n1 and l1
        linkerAnchor1POS = linker_df.loc[linkerAnchorPair[0], ["x", "y", "z"]].astype(float).values
        linker_df["dist2"] = np.sum((linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor1POS) *
                                    (linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor1POS),
                                    axis=1)
        linkerCarbon1Idx = linker_df[linker_df["dist2"] == linker_df[linker_df["el"] == "C"]["dist2"].min()].index.tolist()[0]

        linkerAnchor2POS = linker_df.loc[linkerAnchorPair[1], ["x", "y", "z"]].astype(float).values
        linker_df["dist2"] = np.sum((linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor2POS) *
                                    (linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor2POS),
                                    axis=1)
        linkerCarbon2Idx = linker_df[linker_df["dist2"] == linker_df[linker_df["el"] == "C"]["dist2"].min()].index.tolist()[0]

        linkerAnchorVec = linkerAnchor1POS - linkerAnchor2POS

        rotmat = rotmat2align(-linkerAnchorVec, anchorPairVec)
        linker_df.loc[:, ["x", "y", "z"]] = rotmat.dot(linker_df.loc[:, ["x", "y", "z"]].values.T).T

        displacementVec = node_df.loc[noderAnchorPair[0], ["x", "y", "z"]].astype(float).values - \
            linker_df.loc[linkerCarbon1Idx, ["x", "y", "z"]].astype(float).values
        linker_df.loc[:, ["x", "y", "z"]] = linker_df.loc[:, ["x", "y", "z"]].values + displacementVec

        ldf.append(linker_df[linker_df["el"] != dummyElementCOO].copy(deep=True))
        farAnchorNLPairDispVec = node_df.loc[noderAnchorPair[1], ["x", "y", "z"]].astype(float).values - \
            linker_df.loc[linkerCarbon2Idx, ["x", "y", "z"]].astype(float).values
        latVec.append(farAnchorNLPairDispVec)

    # pandas2xyzfile(node_df, os.path.join(newMOFpath, "node.xyz"))
    final_df = pd.concat(ldf + [node_df[(node_df["el"] != dummyElementCOO) & (node_df["el"] != dummyElementPillar)].copy(deep=True)], axis=0)
    final_df = final_df[final_df["el"] != dummyElementCOO].reset_index(drop=True)[["el", "x", "y", "z"]]
    xyzstr = pandas2xyzfile(final_df, os.path.join(MOFRootdir, "mofCart.xyz"), useStringIO=True)

    mol = mg.Molecule.from_str(xyzstr, fmt="xyz")
    # try:
    #     os.remove(os.path.join(MOFRootdir, "mofCart.xyz"))
    # except:
    #     print(os.path.join(MOFRootdir, "mofCart.xyz") + " does not exist!\n")
    MOFstruct = mg.Structure(np.array(latVec),
                             mol.species,
                             mol.cart_coords,
                             coords_are_cartesian=True)

    df = pd.read_csv("../xyan11-code/OChemDB_bond_threshold.csv", index_col=0)
    element2bondLengthMap = dict(zip(df["element"], df["min"] - (df["stddev"] * 0.01)))
    unique_bond_el = list(set(list(itertools.chain(*[["-".join(sorted([x.symbol, y.symbol])) for x in MOFstruct.species] for y in MOFstruct.species]))))
    unique_bond_el = unique_bond_el + ["Fr-Se"]
    for x in unique_bond_el:
        if x not in element2bondLengthMap:
            element2bondLengthMap[x] = 0.
            for y in x.split("-"):
                if not isinstance(mg.periodic_table.Element(y).atomic_radius_calculated, type(None)):
                    element2bondLengthMap[x] = element2bondLengthMap[x] + mg.periodic_table.Element(y).atomic_radius_calculated
    distMat = MOFstruct.distance_matrix
    distMat[distMat == 0] = np.inf
    bondLengthThresMat = np.array([[element2bondLengthMap["-".join(sorted([x.symbol, y.symbol]))] for x in MOFstruct.species] for y in MOFstruct.species])
    if np.all(distMat > bondLengthThresMat):
        MOFstruct.to(filename=newMOFpath + ".cif", fmt="cif")
        return newMOFpath + ".cif"
    else:
        return ""


def assemble_PillarPaddle_pcuMOF_multiProc(inputDict):
    """multiprocessing wrapper for assemble_PillarPaddle_pcuMOF

    Args:
        inputDict: input of many structures to assemble in the format of dictionary

    Returns:
        returnValue a file path if assembly succeeded, None if failed
    """
    nodePath = inputDict["nodePath"]
    COOLinkerPaths = inputDict["COOLinkerPaths"]
    PillarLinkerPath = inputDict["PillarLinkerPath"]
    newMOFdir = inputDict["newMOFdir"]
    if "dummyElementCOO" in inputDict.keys():
        dummyElementCOO = inputDict["dummyElementCOO"]
    else:
        dummyElementCOO = "At"

    if "dummyElementPillar" in inputDict.keys():
        dummyElementPillar = inputDict["dummyElementPillar"]
    else:
        dummyElementPillar = "Fr"

    returnValue = None
    try:
        returnValue = assemble_pillaredPaddleWheel_pcuMOF(nodePath,
                                                          COOLinkerPaths,
                                                          PillarLinkerPath,
                                                          newMOFdir,
                                                          dummyElementCOO=dummyElementCOO,
                                                          dummyElementPillar=dummyElementPillar)
    except Exception as e:
        print(e)
    return returnValue


def testPillarMOF():
    linker_base = "inferred_linkers/molGAN-batch512-Linkers"
    linker_folders = os.listdir(linker_base)
    # chosen_linker_folders
    clf = [os.path.join(linker_base, x) for x in linker_folders[0:3]]
    COOLinkers = [os.path.join(clf[0], y) for y in os.listdir(clf[0]) if y.endswith(".xyz") and y.startswith(
        "linker-COO")] + [os.path.join(clf[1], y) for y in os.listdir(clf[1]) if y.endswith(".xyz") and y.startswith("linker-COO")]
    PillarLinker = [os.path.join(clf[2], y) for y in os.listdir(clf[2])
                    if y.endswith(".xyz") and y.startswith("linker-") and "COO" not in y][0]

    nodePath = "nodes/zinc_paddle_pillar.xyz"
    COOLinkerPaths = COOLinkers
    PillarLinkerPath = PillarLinker
    comb_name = "L" + "".join(list(reversed(os.path.split(COOLinkers[0])[-1].replace(".xyz", "").replace("linker-", "").split("-")))) + "-" + \
        "L" + "".join(list(reversed(os.path.split(COOLinkers[1])[-1].replace(".xyz", "").replace("linker-", "").split("-")))) + "-" + \
        "L" + "".join(list(reversed(os.path.split(PillarLinker)[-1].replace(".xyz", "").replace("linker-", "").split("-"))))
    newMOFdir = "newMOFs/molGAN-MOF-" + comb_name
    return assemble_pillaredPaddleWheel_pcuMOF(nodePath,
                                               COOLinkerPaths,
                                               PillarLinkerPath,
                                               newMOFdir)


def assemble_mof(nodes: Sequence[NodeDescription], ligands: Sequence[LigandDescription], topology: str) -> MOFRecord:
    """Generate a new MOF from the description of the nodes, ligands and toplogy

    Args:
        nodes: Descriptions of each node
        ligands: Description of the ligands
        topology: Name of the topology

    Returns:
        A new MOF record
    """
    raise NotImplementedError()
