import io
import os
import pandas as pd
import numpy as np
import networkx as nx
import pymatgen.core as mg


def read_P1_cif(cifpath):
    cif_str = None
    with io.open(cifpath) as rf:
        cif_str = rf.read()
    sections = cif_str.split("loop_")
    for sec in sections:
        if "_cell_length" and "_cell_angle" in sec:
            a = float(sec.split("_cell_length_a")[1].split("\n")[0].strip())
            b = float(sec.split("_cell_length_b")[1].split("\n")[0].strip())
            c = float(sec.split("_cell_length_c")[1].split("\n")[0].strip())
            alpha = float(sec.split("_cell_angle_alpha")[1].split("\n")[0].strip())
            beta = float(sec.split("_cell_angle_beta")[1].split("\n")[0].strip())
            gamma = float(sec.split("_cell_angle_gamma")[1].split("\n")[0].strip())
        elif "_atom_site" in sec:
            sec_lines = list(filter(None, sec.split("\n")))
            header = []
            for i in range(0, len(sec_lines)):
                curr_line = sec_lines[i]
                if "_atom" in curr_line or "_site" in curr_line:
                    header.append(curr_line.strip())
                else:
                    # print(header)
                    df = pd.read_csv(io.StringIO("\n".join(sec_lines[i:])), sep=r"\s+", names=header, index_col=None, header=None)
                    break
            break
    df["_atom_site_element"] = ["".join(filter(lambda x: x.isalpha(), y)) for y in df["_atom_site_label"]]
    vc = df["_atom_site_element"].value_counts()
    elements = list(vc.keys())
    df.loc[:, "_atom_site_label"] = df.loc[:, "_atom_site_label"].astype(str)
    for el in elements:
        el_id = df[df["_atom_site_element"]==el].index
        df.loc[el_id, "_atom_site_label"] = [df.loc[el_id[x_i], "_atom_site_element"] + str(x_i+1) for x_i in range(0, len(el_id))]
    df = df[["_atom_site_element", "_atom_site_label", "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]]
    return df, (a, b, c, alpha, beta, gamma), elements

def fragment_single_MOF(cifpath, prep_training_not_assembly=True, visualize=False, allow_metals =["Zn", "Cu", "Zr"]):
    atom_df, lp, el = read_P1_cif(cifpath)
    a, b, c, alpha, beta, gamma = lp
    n2 = (np.cos(alpha * np.pi / 180.) - np.cos(gamma * np.pi / 180.) * np.cos(beta * np.pi / 180.)) / np.sin(gamma * np.pi / 180.)
    M  = np.array([[a,                                0.,                               0.                                                     ],
                   [b * np.cos(gamma * np.pi / 180.), b * np.sin(gamma * np.pi / 180.), 0.                                                     ],
                   [c * np.cos(beta * np.pi / 180.),  c * n2,                           c * np.sqrt(np.sin(beta * np.pi / 180.) ** 2 - n2 ** 2)]])
    
    # proximity bonds detection with PBC considered
    atom_df = pd.concat([atom_df, pd.DataFrame(atom_df.loc[:, ["_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]].values @ M, columns=["x", "y", "z"])], axis=1)
    frac_xyz = atom_df[["_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]].values
    Natoms = frac_xyz.shape[0]
    frac_diff = frac_xyz[:, np.newaxis, :] - frac_xyz[np.newaxis, :, :]
    frac_diff[frac_diff > 0.5] -= 1.0
    frac_diff[frac_diff < -0.5] += 1.0
    frac_diff = frac_diff.reshape(Natoms**2, 3)
    cart_diff = frac_diff @ M
    cart_dist_sqr = np.sum(cart_diff * cart_diff, axis=1)
    cart_dist_sqr_mat = cart_dist_sqr.reshape(Natoms, Natoms)
    
    thres_df = pd.read_csv("OChemDB_bond_threshold.csv", index_col=0)
    element2bondLengthMap = dict(zip(thres_df["element"], thres_df["max"] + (thres_df["stddev"] * 0.1)))
    el_list = atom_df["_atom_site_element"].to_list()
    unique_el_list = list(set(el_list))
    unique_bond_el = list(set(["-".join(sorted([x, y])) for x in unique_el_list for y in unique_el_list]))
    unique_bond_el = unique_bond_el + ["Fr-Se"]
    for x in unique_bond_el:
        if x not in element2bondLengthMap:
            element2bondLengthMap[x] = 0.
            for y in x.split("-"):
                if not isinstance(mg.periodic_table.Element(y).atomic_radius_calculated, type(None)):
                    element2bondLengthMap[x] = element2bondLengthMap[x] + mg.periodic_table.Element(y).atomic_radius_calculated
    bondLengthThresMat = np.array([[element2bondLengthMap["-".join(sorted([x, y]))] for x in el_list] for y in el_list])
    np.fill_diagonal(bondLengthThresMat, 0.)
    bondLengthThresMat_sqr = bondLengthThresMat * bondLengthThresMat
    adjmat = pd.DataFrame(bondLengthThresMat_sqr > cart_dist_sqr_mat)
    edge_list = list(set([tuple(sorted(x)) for x in np.array(np.where(adjmat)).T.tolist()]))
    
    bond_df = pd.DataFrame(edge_list, columns=["atom1", "atom2"])
    bond_df["el1"] = bond_df["atom1"].map(dict(zip(atom_df.index, el_list)))
    bond_df["el2"] = bond_df["atom2"].map(dict(zip(atom_df.index, el_list)))
    
    # find the bonds that need to be cut
    metal_N_bond_ids = None
    if "N" in unique_el_list:
        metal_N_bond_ids = bond_df[((bond_df["el1"]=="N")&(bond_df["el2"].isin(allow_metals)))|((bond_df["el2"]=="N")&(bond_df["el1"].isin(allow_metals)))].index
    metal_O_bond_ids = None
    if "O" in unique_el_list:
        metal_O_bond_ids = bond_df[((bond_df["el1"]=="O")&(bond_df["el2"].isin(allow_metals)))|((bond_df["el2"]=="O")&(bond_df["el1"].isin(allow_metals)))].index
    
    MN = bond_df.loc[metal_N_bond_ids, :]
    MN_atom_ids = list(set(MN["atom1"].to_list() + MN["atom2"].to_list()))
    MN_atoms = atom_df.loc[MN_atom_ids, :]
    N_ids = MN_atoms[MN_atoms["_atom_site_element"]=="N"].index.tolist()
    CN3_bond_ids = bond_df[((bond_df["atom1"].isin(N_ids))&(bond_df["el2"]=="C"))|((bond_df["atom2"].isin(N_ids))&(bond_df["el1"]=="C"))].index
    CN3_bond = bond_df.loc[CN3_bond_ids, :].copy(deep=True).reset_index(drop=True)
    CN3_bond["C"] = None
    CN3_bond["N"] = None
    for cn3_idx in CN3_bond.index:
        if CN3_bond.at[cn3_idx, "el1"] == "N":
            CN3_bond.at[cn3_idx, "C"] = CN3_bond.at[cn3_idx, "atom2"]
            CN3_bond.at[cn3_idx, "N"] = CN3_bond.at[cn3_idx, "atom1"]
        else:
            CN3_bond.at[cn3_idx, "C"] = CN3_bond.at[cn3_idx, "atom1"]
            CN3_bond.at[cn3_idx, "N"] = CN3_bond.at[cn3_idx, "atom2"]
    find_C_by_N = dict(zip(CN3_bond["N"].to_list(), CN3_bond["C"].to_list()))
    
    MO = bond_df.loc[metal_O_bond_ids, :]
    MO_atom_ids = list(set(MO["atom1"].to_list() + MO["atom2"].to_list()))
    MO_atoms = atom_df.loc[MO_atom_ids, :]
    O_ids = MO_atoms[MO_atoms["_atom_site_element"]=="O"].index.tolist()
    CO_bond_ids = bond_df[(bond_df["atom1"].isin(O_ids))&(bond_df["el2"]=="C")|(bond_df["atom2"].isin(O_ids))&(bond_df["el1"]=="C")].index
    CO_bonds = bond_df.loc[CO_bond_ids, :]
    C_ids = list(set(CO_bonds["atom1"].to_list() + CO_bonds["atom2"].to_list()) - set(O_ids))
    find_O_by_C = [[cid, list(set(CO_bonds[((CO_bonds["atom1"]==cid)&(CO_bonds["el2"]=="O"))|((CO_bonds["atom2"]==cid)&(CO_bonds["el1"]=="O"))][["atom1", "atom2"]].values.flatten().tolist()) - {cid})] for cid in C_ids]
    find_O_by_C = dict(zip(*list(map(list, zip(*find_O_by_C)))))
    CC_bond_ids = bond_df[((bond_df["atom1"].isin(C_ids))&(~bond_df["atom2"].isin(O_ids)))|((bond_df["atom2"].isin(C_ids))&(~bond_df["atom1"].isin(O_ids)))].index
    
    edges2remove = bond_df.loc[CC_bond_ids.tolist() + metal_N_bond_ids.tolist(), :]
    edges2remove = edges2remove[["atom1", "atom2"]].values.tolist()
    
    # use networkx to make graphs
    G = nx.from_pandas_adjacency(adjmat)
    atom_df["id"] = atom_df.index
    attrs = dict(zip(atom_df.index, atom_df.to_dict(orient="records")))
    nx.set_node_attributes(G, attrs)
    nx.set_node_attributes(G, 0, "anchor_atom_mask")
    
    labels = nx.get_node_attributes(G, '_atom_site_element')
    if visualize:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        nx.draw(G, pos=nx.spring_layout(G), labels=labels, node_size=200)
        fig.savefig("mof-" + ciffile + ".png", dpi=300)
    
    # iterate through all targeted bonds, remove bonds, add dummy atoms, add connections to the subgraphs
    anchor_counter = 0
    for e2r in edges2remove:
        anchor_counter = anchor_counter - 1
        G.remove_edge(e2r[0], e2r[1])
        new_id_e2r0 = max(list(G.nodes))+1
        new_id_e2r1 = max(list(G.nodes))+2
        attr_e2r0 = dict(G.nodes[e2r[0]])
        attr_e2r0["id"] = new_id_e2r0
        attr_e2r1 = dict(G.nodes[e2r[1]])
        attr_e2r1["id"] = new_id_e2r1
        # C#N-M treatment
        if (G.nodes[e2r[0]]["_atom_site_element"] == "N" and G.nodes[e2r[1]]["_atom_site_element"] in allow_metals) or \
            (G.nodes[e2r[1]]["_atom_site_element"] == "N" and G.nodes[e2r[0]]["_atom_site_element"] in allow_metals):
            if G.nodes[e2r[0]]["_atom_site_element"] in allow_metals:
                attr_e2r1["_atom_site_element"] = "Fr"
                attr_e2r1["anchor_atom_mask"] = anchor_counter
                G.add_node(new_id_e2r1)
                nx.set_node_attributes(G, {new_id_e2r1: attr_e2r1})
                G.add_edge(new_id_e2r1, e2r[0])
                G.nodes[e2r[1]]["anchor_atom_mask"] = anchor_counter
                G.nodes[find_C_by_N[e2r[1]]]["anchor_atom_mask"] = anchor_counter
                if not prep_training_not_assembly:
                    attr_e2r0["_atom_site_element"] = "Fr"
                    G.add_node(new_id_e2r0)
                    nx.set_node_attributes(G, {new_id_e2r0: attr_e2r0})
                    G.add_edge(new_id_e2r0, e2r[1])
            elif G.nodes[e2r[1]]["_atom_site_element"] in allow_metals:
                attr_e2r0["_atom_site_element"] = "Fr"
                attr_e2r0["anchor_atom_mask"] = anchor_counter
                G.add_node(new_id_e2r0)
                nx.set_node_attributes(G, {new_id_e2r0: attr_e2r0})
                G.add_edge(new_id_e2r0, e2r[1])
                G.nodes[e2r[0]]["anchor_atom_mask"] = anchor_counter
                G.nodes[find_C_by_N[e2r[0]]]["anchor_atom_mask"] = anchor_counter
                if not prep_training_not_assembly:
                    attr_e2r1["_atom_site_element"] = "Fr"
                    G.add_node(new_id_e2r1)
                    nx.set_node_attributes(G, {new_id_e2r1: attr_e2r1})
                    G.add_edge(new_id_e2r1, e2r[0])
    
        # COO-M treatment
        elif G.nodes[e2r[0]]["_atom_site_element"] == "C" and G.nodes[e2r[1]]["_atom_site_element"] == "C":
            if e2r[0] in find_O_by_C.keys():
                COO_C_id = e2r[0]
                _COO_C_id = new_id_e2r0
                node_anchor = new_id_e2r1
                ligand_anchor = new_id_e2r0
            else:
                COO_C_id = e2r[1]
                _COO_C_id = new_id_e2r1
                node_anchor = new_id_e2r0
                ligand_anchor = new_id_e2r1
    
            # find related oyxgen atoms used for training dataset
            if prep_training_not_assembly:
                COO_O0, COO_O1 = find_O_by_C[COO_C_id]
                new_id_COO_O0 = max(list(G.nodes))+3
                new_id_COO_O1 = max(list(G.nodes))+4
                attr_COO_O0 = dict(G.nodes[COO_O0])
                attr_COO_O0["id"] = new_id_COO_O0
                attr_COO_O1 = dict(G.nodes[COO_O1])
                attr_COO_O1["id"] = new_id_COO_O1
                G.add_node(new_id_COO_O0)
                G.add_node(new_id_COO_O1)
                attr_COO_O0["anchor_atom_mask"] = anchor_counter
                attr_COO_O1["anchor_atom_mask"] = anchor_counter
                nx.set_node_attributes(G, {new_id_COO_O0: attr_COO_O0})
                nx.set_node_attributes(G, {new_id_COO_O1: attr_COO_O1})
                G.add_edge(new_id_COO_O0, _COO_C_id)
                G.add_edge(new_id_COO_O1, _COO_C_id)
            else:
                attr_e2r0["_atom_site_element"] = "At"
                attr_e2r1["_atom_site_element"] = "At"
            G.add_node(new_id_e2r0)
            nx.set_node_attributes(G, {new_id_e2r0: attr_e2r0})
            G.add_node(new_id_e2r1)
            nx.set_node_attributes(G, {new_id_e2r1: attr_e2r1})
            G.nodes[node_anchor]["_atom_site_element"] = "At"
            if prep_training_not_assembly:
                G.nodes[ligand_anchor]["anchor_atom_mask"] = anchor_counter
                G.nodes[node_anchor]["anchor_atom_mask"] = anchor_counter
            G.add_edge(new_id_e2r0, e2r[1])
            G.add_edge(new_id_e2r1, e2r[0])
    
    # convert connected components to xyz
    xyz_strs = []
    cc = list(nx.connected_components(G))
    for i in range(0, len(cc)):
        c = cc[i]
        subG = G.subgraph(c).copy()
        xyz_df = pd.DataFrame.from_dict([subG.nodes[x] for x in subG.nodes])[["_atom_site_element", "x", "y", "z", "anchor_atom_mask"]].sort_values(
            by=["anchor_atom_mask", "_atom_site_element"], ascending=True).reset_index(drop=True)
        xyz_df["anchor_atom_mask"] = -xyz_df["anchor_atom_mask"]
        if len(list(set(xyz_df["_atom_site_element"].to_list()).intersection(set(allow_metals)))) == 0:
            # ligand
            print("organic Ligand")
            # prep for training
        else:
            # metal node
            print("metal Node")
        xyz_str = str(len(xyz_df)) + "\n\n" + xyz_df[["_atom_site_element", "x", "y", "z"]].to_string(header=None, index=None)
        xyz_strs.append(xyz_str)
        if visualize:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            labels = nx.get_node_attributes(subG, '_atom_site_element')
            nx.draw(subG, pos=nx.spring_layout(subG), labels=labels, node_size=200, ax=ax)
            fig.savefig("sbu-" + str(i) + ".png", dpi=300)
    return xyz_strs
