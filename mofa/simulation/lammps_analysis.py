import multiprocessing
import numpy as np
import pandas as pd
import os
import io
import matplotlib.pyplot as plt


def run_analysis(input_dict):
    """analyze the lattice parameters of MOF during LAMMPS NPT simulation

    Args:
        input_dict: a dictionary with necessary inputs
            lmp_job_path: lammps job directory
    Returns:
        ret_dict: a dictionary with output values
    """
    dirname = input_dict["lmp_job_path"]
    read_str = None
    with io.open(os.path.join(dirname, "log.lammps")) as rf:
        read_str = rf.read()
    df = pd.read_csv(io.StringIO(read_str.split("bytes")[1].split("Loop time")[0]), sep=r"\s+", header=0)
    ret_dict = {"dirname": input_dict["lmp_job_path"]}

    # plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    delta = df["Cella"].mean() - df["Cella"].to_list()[0]
    delta_rel = delta / df["Cella"].to_list()[0]

    ret_dict["a_mean"] = df["Cella"].mean()
    ret_dict["a_std"] = df["Cella"].std()
    ret_dict["a_delta"] = delta
    ret_dict["a_delta_rel"] = delta_rel

    ax.plot(df["Time"] / 1000000., df["Cella"], lw=0.5,
            label=r"$a$" + " = " + "%.3f" % df["Cella"].mean() +
            "±" + "%.4f" % df["Cella"].std() + ", " +
            r"$\bar{a} - a_0$" + " = " + "%.3f" % delta +
            " (" + "%+.1f" % (delta_rel * 100) + "%)")
    delta = df["Cellb"].mean() - df["Cellb"].to_list()[0]
    delta_rel = delta / df["Cellb"].to_list()[0]

    ret_dict["b_mean"] = df["Cellb"].mean()
    ret_dict["b_std"] = df["Cellb"].std()
    ret_dict["b_delta"] = delta
    ret_dict["b_delta_rel"] = delta_rel

    ax.plot(df["Time"] / 1000000., df["Cellb"], lw=0.5,
            label=r"$b$" + " = " + "%.3f" % df["Cellb"].mean() +
            "±" + "%.4f" % df["Cellb"].std() + ", " +
            r"$\bar{b} - b_0$" + " = " + "%.3f" % delta +
            " (" + "%+.1f" % (delta_rel * 100) + "%)")
    delta = df["Cellc"].mean() - df["Cellc"].to_list()[0]
    delta_rel = delta / df["Cellc"].to_list()[0]

    ret_dict["c_mean"] = df["Cellc"].mean()
    ret_dict["c_std"] = df["Cellc"].std()
    ret_dict["c_delta"] = delta
    ret_dict["c_delta_rel"] = delta_rel

    ax.plot(df["Time"] / 1000000., df["Cellc"], lw=0.5,
            label=r"$b$" + " = " + "%.3f" % df["Cellc"].mean() +
            "±" + "%.4f" % df["Cellc"].std() + ", " +
            r"$\bar{c} - c_0$" + " = " + "%.3f" % delta +
            " (" + "%+.1f" % (delta_rel * 100) + "%)")

    ax.set_ylabel("Lattice Vector Length (Å)")
    ax.set_xlabel("Simulation Time (ns)")
    ax.legend()
    ax.grid()
    fig.savefig(os.path.join(dirname, "length.png"), dpi=300, transparent=True)
    plt.close('all')

    fig, ax = plt.subplots(figsize=(6, 4))
    delta = df["CellAlpha"].mean() - df["CellAlpha"].to_list()[0]
    delta_rel = delta / df["CellAlpha"].to_list()[0]

    ret_dict["alpha_mean"] = df["CellAlpha"].mean()
    ret_dict["alpha_std"] = df["CellAlpha"].std()
    ret_dict["alpha_delta"] = delta
    ret_dict["alpha_delta_rel"] = delta_rel

    ax.plot(df["Time"] / 1000000., df["CellAlpha"], lw=0.5,
            label=r"$\alpha$" + " = " + "%.3f" % df["CellAlpha"].mean() +
            "±" + "%.4f" % df["CellAlpha"].std() + ", " +
            r"$\bar{\alpha} - \alpha_0$" + " = " + "%.3f" % delta +
            " (" + "%+.1f" % (delta_rel * 100) + "%)")
    delta = df["CellBeta"].mean() - df["CellBeta"].to_list()[0]
    delta_rel = delta / df["CellBeta"].to_list()[0]

    ret_dict["beta_mean"] = df["CellBeta"].mean()
    ret_dict["beta_std"] = df["CellBeta"].std()
    ret_dict["beta_delta"] = delta
    ret_dict["beta_delta_rel"] = delta_rel

    ax.plot(df["Time"] / 1000000., df["CellBeta"], lw=0.5,
            label=r"$\beta$" + " = " + "%.3f" % df["CellBeta"].mean() +
            "±" + "%.4f" % df["CellBeta"].std() + ", " +
            r"$\bar{\beta} - \beta_0$" + " = " + "%.3f" % delta +
            " (" + "%+.1f" % (delta_rel * 100) + "%)")
    delta = df["CellGamma"].mean() - df["CellGamma"].to_list()[0]
    delta_rel = delta / df["CellGamma"].to_list()[0]

    ret_dict["gamma_mean"] = df["CellGamma"].mean()
    ret_dict["gamma_std"] = df["CellGamma"].std()
    ret_dict["gamma_delta"] = delta
    ret_dict["gamma_delta_rel"] = delta_rel

    ax.plot(df["Time"] / 1000000., df["CellGamma"], lw=0.5,
            label=r"$\gamma$" + " = " + "%.3f" % df["CellGamma"].mean() +
            "±" + "%.4f" % df["CellGamma"].std() + ", " +
            r"$\bar{\gamma} - \gamma_0$" + " = " + "%.3f" % delta +
            " (" + "%+.1f" % (delta_rel * 100) + "%)")

    ax.set_ylabel("Lattice Vector Angle (°)")
    ax.set_xlabel("Simulation Time (ns)")
    ax.legend()
    ax.grid()
    fig.savefig(os.path.join(dirname, "angle.png"), dpi=300, transparent=True)
    plt.close('all')
    # df[['Cella', 'Cellb', 'Cellc', 'CellAlpha', 'CellBeta', 'CellGamma']].plot()

    df["rad_alpha"] = df["CellAlpha"] * np.pi / 180
    df["rad_beta"] = df["CellBeta"] * np.pi / 180
    df["rad_gamma"] = df["CellGamma"] * np.pi / 180

    df["Volume"] = df["Cella"] * df["Cellb"] * df["Cellc"] * np.sqrt(
        np.abs(
            1 + (2 * np.cos(df["rad_alpha"]) * np.cos(df["rad_beta"]) * np.cos(df["rad_gamma"]))
            - (np.cos(df["rad_alpha"]) * np.cos(df["rad_alpha"]))
            - (np.cos(df["rad_beta"]) * np.cos(df["rad_beta"]))
            - (np.cos(df["rad_gamma"]) * np.cos(df["rad_gamma"]))
        )
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax2 = ax.twinx()

    delta = df["Density"].mean() - df["Density"].to_list()[0]
    delta_rel = delta / df["Density"].to_list()[0]

    ret_dict["den_mean"] = df["Density"].mean()
    ret_dict["den_std"] = df["Density"].std()
    ret_dict["den_delta"] = delta
    ret_dict["den_delta_rel"] = delta_rel

    ax2.plot(df["Time"] / 1000000., df["Density"], lw=0.5,
             color="tab:purple",
             label="%.2f" % df["Density"].mean() +
             "±" + "%.3f" % df["Density"].std() + ", " +
             r"$\bar{\rho}-\rho_0$" + " = " + "%.3f" % delta +
             " (" + "%+.1f" % (delta_rel * 100) + "%)")
    ax.plot(df["Time"] / 1000000., df["Volume"] / 1000, lw=0.5,
            color="tab:purple",
            label="%.2f" % df["Density"].mean() +
            "±" + "%.3f" % df["Density"].std() + ", " +
            r"$\bar{\rho}-\rho_0$" + " = " + "%.3f" % delta +
            " (" + "%+.1f" % (delta_rel * 100) + "%)")

    ax2.set_ylabel("Density (g/cm" + r"$^3$" + ")")

    delta = (df["Volume"].mean() - df["Volume"].to_list()[0]) / 1000
    delta_rel = delta / df["Volume"].to_list()[0]

    ret_dict["vol_mean"] = df["Volume"].mean()
    ret_dict["vol_std"] = df["Volume"].std()
    ret_dict["vol_delta"] = delta
    ret_dict["vol_delta_rel"] = delta_rel

    ax.plot(df["Time"] / 1000000., df["Volume"] / 1000, lw=0.5,
            color="tab:blue",
            label="%.2f" % df["Volume"].mean() +
            "±" + "%.3f" % df["Volume"].std() + ", " +
            r"$\bar{V}-V_0$" + " = " + "%.3f" % delta +
            " (" + "%+.1f" % (delta_rel * 100) + "%)")

    ax.set_ylabel("Volume (nm" + r"$^3$" + ")")
    ax.set_xlabel("Simulation Time (ns)")

    ax.spines['left'].set_color('tab:blue')
    ax.spines['right'].set_color('tab:purple')
    ax.yaxis.label.set_color('tab:blue')
    ax2.yaxis.label.set_color('tab:purple')
    ax.tick_params(axis='y', colors='tab:blue')
    ax2.tick_params(axis='y', colors='tab:purple')

    ax.legend(loc=5)
    ax.grid()
    fig.savefig(os.path.join(dirname, "density_volume.png"), dpi=300, transparent=True)
    plt.close('all')

    df.to_csv(os.path.join(dirname, "log.csv"))
    return ret_dict


df = pd.read_csv("ghp_ge2-std.csv", index_col=0)
lmp_dir = "new_lmp_sim"
lmp_jobs = [os.path.join(lmp_dir, x) for x in df["name"].to_list() if os.path.isdir(os.path.join(lmp_dir, x))]
input_dicts = [{"lmp_job_path": lmp_job} for lmp_job in lmp_jobs if "relaxing.400000.restart" in os.listdir(lmp_job)]

Npool = int(8)
print("Createing CPU pool with " + str(Npool) + " slots for analysis...\n\n")

ret_dicts = None
with multiprocessing.Pool(Npool) as mpool:
    ret_dicts = mpool.map_async(run_analysis, input_dicts).get()
mpool.close()
result = pd.DataFrame(ret_dicts)
result.to_csv("./lmp_summary.csv")
