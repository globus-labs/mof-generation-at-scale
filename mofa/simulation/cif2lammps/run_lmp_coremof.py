import re, sys, os, io, shutil, warnings, subprocess, asyncio
import pandas as pd
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
#from ray.util import multiprocessing
import matplotlib.pyplot as plt
from main_conversion import single_conversion

def preprocessing(cif_name):
    cif_dir = "../coremof_structure_10143"
    cif_path = os.path.join(cif_dir, cif_name)
    lmp_dir = cif_dir + "_lmp"
    lmp_path = os.path.join(lmp_dir, cif_name.replace(".cif", ""))
    os.makedirs(lmp_path, exist_ok=True)
    from UFF4MOF_construction import UFF4MOF
    print("\n\n")
    print("\n\n")
    print(lmp_path)
    try:
        single_conversion(cif_path, 
            force_field=UFF4MOF, 
            ff_string='UFF4MOF', 
            small_molecule_force_field=None, 
            outdir=lmp_path, 
            charges=False, 
            parallel=False, 
            replication='2x2x2', 
            read_cifs_pymatgen=True, 
            add_molecule=None, 
            small_molecule_file=None)
        in_file_name = [x for x in os.listdir(lmp_path) if x.startswith("in.") and not x.startswith("in.lmp")][0]
        data_file_name = [x for x in os.listdir(lmp_path) if x.startswith("data.") and not x.startswith("data.lmp")][0]
        in_file_rename = "in.lmp"
        data_file_rename = "data.lmp"
        print("Reading data file for element list: " + os.path.join(lmp_path, data_file_name))
        with io.open(os.path.join(lmp_path, data_file_name), "r") as rf:
            df = pd.read_csv(io.StringIO(rf.read().split("Masses")[1].split("Pair Coeffs")[0]), sep=r"\s+", header=None)
            element_list = df[3].to_list()
        
        
        with io.open(os.path.join(lmp_path, in_file_rename), "w") as wf:
            print("Writing input file: " + os.path.join(lmp_path, in_file_rename))
            with io.open(os.path.join(lmp_path, in_file_name), "r") as rf:
                print("Reading original input file: " + os.path.join(lmp_path, in_file_name))
                wf.write(rf.read().replace(data_file_name, data_file_rename) + """

# simulation

fix             fxnpt all npt temp 300.0 300.0 100.0 tri 1.0 1.0 800.0
variable        Nevery equal 1000

thermo          ${Nevery}
thermo_style    custom step cpu dt time temp press pe ke etotal density xlo xhi ylo yhi zlo zhi cella cellb cellc cellalpha cellbeta cellgamma
thermo_modify   flush yes

minimize        1.0e-10 1.0e-10 10000 1000000
reset_timestep  0

dump            trajectAll all custom ${Nevery} dump.lammpstrj.all.0 id type element x y z q
dump_modify     trajectAll element """ + " ".join(element_list) + """


timestep        0.5
run             10000
timestep        1.0
run             200000
undump          trajectAll
write_restart   relaxing.*.restart
write_data      relaxing.*.data

""")
        os.remove(os.path.join(lmp_path, in_file_name))
        shutil.move(os.path.join(lmp_path, data_file_name), os.path.join(lmp_path, data_file_rename))
        print("Success!!\n\n")
        print("###############################################")
        print("###############################################")
        print("###############################################")
        print("###############################################")
        print("\n\n")
        print("\n\n")

    except Exception as e:
        print(e)
        shutil.rmtree(lmp_path)

def run_lmp_simulation(input_dict):
    Ncpus_per_job = input_dict["Ncpus_per_job"]
    lmp_job_path = input_dict["lmp_job_path"]
    #if "relaxing.2100000.restart" not in os.listdir(lmp_job_path):
    #print("######\nRunning LAMMPS simulation in " + lmp_job_path + " with " + str(Ncpus_per_job) + " CPUs...\n######\n\n")
    # keep the "--bind-to none" in mpirun to ensure concurrent execution!!!
    CompletedProcess = subprocess.run("cd " + lmp_job_path + " && mpirun -np " + str(int(Ncpus_per_job)) + " --bind-to none lmp_mpi -in in.lmp &", 
                                        shell=True, capture_output=True)
    # if os.path.exists(os.path.join(lmp_job_path, "nohup.out")):
    #     os.remove(os.path.join(lmp_job_path, "nohup.out"))
    with io.open(os.path.join(lmp_job_path, "stdout.txt"), "w", newline="\n") as wf:
        wf.write(str(CompletedProcess.stdout, 'UTF-8'))
    with io.open(os.path.join(lmp_job_path, "stderr.txt"), "w", newline="\n") as wf:
        wf.write(str(CompletedProcess.stderr, 'UTF-8'))
    #print("$$$$$$\nLAMMPS simulation exited in " + lmp_job_path + "\n$$$$$$\n\n")
    return

def run_analysis(input_dict):
    dirname = input_dict["lmp_job_path"]
    #print(dirname)
    read_str = None
    with io.open(os.path.join(dirname, "log.lammps")) as rf:
        read_str = rf.read()
        
    df_strs = [x.split("Loop time")[0] for x in read_str.split("bytes")[1:]]
    df_list = []
    for df_str in df_strs:
        lines_list = df_str.strip().split("\n")
        fixed_str_list = []
        header = list(filter(None, lines_list[0].split(" ")))
        line_i = 1
        while line_i < len(lines_list):
            orig_line = lines_list[line_i]
            if orig_line.startswith("WARNING"):
                line_i = line_i + 1
                continue
            row_1D = np.array(list(filter(None, orig_line.split(" "))))
            #print(len(row_1D.tolist()), len(header))
            while len(row_1D.tolist()) % len(header) != 0:
                line_i = line_i + 1
                orig_line = orig_line + "\n" + lines_list[line_i]
                row_1D = np.array(list(filter(None, orig_line.split(" "))))
                #print(len(row_1D.tolist()), len(header), len(row_1D.tolist()) % len(header))
            rows = row_1D.reshape(int(len(row_1D) / len(header)), len(header))
            fixed_str_list.append("\n".join([" ".join(x) for x in rows.tolist()]))
            line_i = line_i + 1
        fixed_str = "\n".join(fixed_str_list)
        df = pd.read_csv(io.StringIO(fixed_str), sep=" ", header=None)
        df.columns = header
        df_list.append(df)

    ret_dict = {"dirname": input_dict["lmp_job_path"]}
    #return ret_dict
    df = pd.concat([df_list[1], df_list[2].loc[1:, :]], axis=0).reset_index(drop=True)

    # plotting
    fig, ax = plt.subplots(figsize=(6,4))
    delta = df["Cella"].mean() - df["Cella"].to_list()[0]
    delta_rel = delta / df["Cella"].to_list()[0]

    ret_dict["a_mean"] = df["Cella"].mean()
    ret_dict["a_std"] = df["Cella"].std()
    ret_dict["a_delta"] = delta
    ret_dict["a_delta_rel"] = delta_rel
    
    ax.plot(df["Time"] / 1000000., df["Cella"], lw=0.5,
            label=r"$a$" + " = " + "%.3f" % df["Cella"].mean() + \
                "±" + "%.4f" % df["Cella"].std() + ", " + \
                r"$\bar{a} - a_0$" + " = " + "%.3f" % delta + \
                " (" + "%+.1f" % (delta_rel*100) + "%)")
    delta = df["Cellb"].mean() - df["Cellb"].to_list()[0]
    delta_rel = delta / df["Cellb"].to_list()[0]
    
    ret_dict["b_mean"] = df["Cellb"].mean()
    ret_dict["b_std"] = df["Cellb"].std()
    ret_dict["b_delta"] = delta
    ret_dict["b_delta_rel"] = delta_rel

    ax.plot(df["Time"] / 1000000., df["Cellb"], lw=0.5,
            label=r"$b$" + " = " + "%.3f" % df["Cellb"].mean() + \
                "±" + "%.4f" % df["Cellb"].std() + ", " + \
                r"$\bar{b} - b_0$" + " = " + "%.3f" % delta + \
                " (" + "%+.1f" % (delta_rel*100) + "%)")
    delta = df["Cellc"].mean() - df["Cellc"].to_list()[0]
    delta_rel = delta / df["Cellc"].to_list()[0]
    
    ret_dict["c_mean"] = df["Cellc"].mean()
    ret_dict["c_std"] = df["Cellc"].std()
    ret_dict["c_delta"] = delta
    ret_dict["c_delta_rel"] = delta_rel

    ax.plot(df["Time"] / 1000000., df["Cellc"], lw=0.5,
            label=r"$b$" + " = " + "%.3f" % df["Cellc"].mean() + \
                "±" + "%.4f" % df["Cellc"].std() + ", " + \
                r"$\bar{c} - c_0$" + " = " + "%.3f" % delta + \
                " (" + "%+.1f" % (delta_rel*100) + "%)")

    ax.set_ylabel("Lattice Vector Length (Å)")
    ax.set_xlabel("Simulation Time (ns)")
    ax.legend()
    ax.grid()
    fig.savefig(os.path.join(dirname, "length.png"), dpi=300, transparent=True)
    plt.close('all')

    fig, ax = plt.subplots(figsize=(6,4))
    delta = df["CellAlpha"].mean() - df["CellAlpha"].to_list()[0]
    delta_rel = delta / df["CellAlpha"].to_list()[0]
    
    ret_dict["alpha_mean"] = df["CellAlpha"].mean()
    ret_dict["alpha_std"] = df["CellAlpha"].std()
    ret_dict["alpha_delta"] = delta
    ret_dict["alpha_delta_rel"] = delta_rel

    ax.plot(df["Time"] / 1000000., df["CellAlpha"], lw=0.5,
            label=r"$\alpha$" + " = " + "%.3f" % df["CellAlpha"].mean() + \
                "±" + "%.4f" % df["CellAlpha"].std() + ", " + \
                r"$\bar{\alpha} - \alpha_0$" + " = " + "%.3f" % delta + \
                " (" + "%+.1f" % (delta_rel*100) + "%)")
    delta = df["CellBeta"].mean() - df["CellBeta"].to_list()[0]
    delta_rel = delta / df["CellBeta"].to_list()[0]
    
    ret_dict["beta_mean"] = df["CellBeta"].mean()
    ret_dict["beta_std"] = df["CellBeta"].std()
    ret_dict["beta_delta"] = delta
    ret_dict["beta_delta_rel"] = delta_rel

    ax.plot(df["Time"] / 1000000., df["CellBeta"], lw=0.5,
            label=r"$\beta$" + " = " + "%.3f" % df["CellBeta"].mean() + \
                "±" + "%.4f" % df["CellBeta"].std() + ", " + \
                r"$\bar{\beta} - \beta_0$" + " = " + "%.3f" % delta + \
                " (" + "%+.1f" % (delta_rel*100) + "%)")
    delta = df["CellGamma"].mean() - df["CellGamma"].to_list()[0]
    delta_rel = delta / df["CellGamma"].to_list()[0]
    
    ret_dict["gamma_mean"] = df["CellGamma"].mean()
    ret_dict["gamma_std"] = df["CellGamma"].std()
    ret_dict["gamma_delta"] = delta
    ret_dict["gamma_delta_rel"] = delta_rel

    ax.plot(df["Time"] / 1000000., df["CellGamma"], lw=0.5,
            label=r"$\gamma$" + " = " + "%.3f" % df["CellGamma"].mean() + \
                "±" + "%.4f" % df["CellGamma"].std() + ", " + \
                r"$\bar{\gamma} - \gamma_0$" + " = " + "%.3f" % delta + \
                " (" + "%+.1f" % (delta_rel*100) + "%)")

    ax.set_ylabel("Lattice Vector Angle (°)")
    ax.set_xlabel("Simulation Time (ns)")
    ax.legend()
    ax.grid()
    fig.savefig(os.path.join(dirname, "angle.png"), dpi=300, transparent=True)
    plt.close('all')
    #df[['Cella', 'Cellb', 'Cellc', 'CellAlpha', 'CellBeta', 'CellGamma']].plot()

    df["rad_alpha"] = df["CellAlpha"] * np.pi / 180
    df["rad_beta"] = df["CellBeta"] * np.pi / 180
    df["rad_gamma"] = df["CellGamma"] * np.pi / 180

    df["Volume"] = df["Cella"] * df["Cellb"] * df["Cellc"] * np.sqrt(
        np.abs(
            1 + (2 * np.cos(df["rad_alpha"]) * np.cos(df["rad_beta"]) * np.cos(df["rad_gamma"])) \
            - (np.cos(df["rad_alpha"]) * np.cos(df["rad_alpha"])) \
            - (np.cos(df["rad_beta"]) * np.cos(df["rad_beta"])) \
            - (np.cos(df["rad_gamma"]) * np.cos(df["rad_gamma"]))
        )
    )
    fig, ax = plt.subplots(figsize=(6,4))
    ax2 = ax.twinx()

    delta = df["Density"].mean() - df["Density"].to_list()[0]
    delta_rel = delta / df["Density"].to_list()[0]
    
    ret_dict["den_mean"] = df["Density"].mean()
    ret_dict["den_std"] = df["Density"].std()
    ret_dict["den_delta"] = delta
    ret_dict["den_delta_rel"] = delta_rel

    ax2.plot(df["Time"] / 1000000., df["Density"], lw=0.5,
            color="tab:purple",
            label= "%.2f" % df["Density"].mean() + \
                "±" + "%.3f" % df["Density"].std() + ", " + \
                r"$\bar{\rho}-\rho_0$" + " = " + "%.3f" % delta + \
                " (" + "%+.1f" % (delta_rel*100) + "%)")
    ax.plot(df["Time"] / 1000000., df["Volume"] / 1000, lw=0.5,
            color="tab:purple",
            label= "%.2f" % df["Density"].mean() + \
                "±" + "%.3f" % df["Density"].std() + ", " + \
                r"$\bar{\rho}-\rho_0$" + " = " + "%.3f" % delta + \
                " (" + "%+.1f" % (delta_rel*100) + "%)")

    ax2.set_ylabel("Density (g/cm" + r"$^3$" + ")")

    delta = (df["Volume"].mean() - df["Volume"].to_list()[0]) / 1000
    delta_rel = delta / df["Volume"].to_list()[0]
    
    ret_dict["vol_mean"] = df["Volume"].mean()
    ret_dict["vol_std"] = df["Volume"].std()
    ret_dict["vol_delta"] = delta
    ret_dict["vol_delta_rel"] = delta_rel

    ax.plot(df["Time"] / 1000000., df["Volume"] / 1000, lw=0.5,
            color="tab:blue",
            label= "%.2f" % df["Volume"].mean() + \
                "±" + "%.3f" % df["Volume"].std() + ", " + \
                r"$\bar{V}-V_0$" + " = " + "%.3f" % delta + \
                " (" + "%+.1f" % (delta_rel*100) + "%)")

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

if __name__ == '__main__':
    # do stuff that takes long time
    
    warnings.filterwarnings("ignore")
    cif_dir = "../coremof_structure_10143"
    lmp_dir = cif_dir + "_lmp"
    os.makedirs(lmp_dir, exist_ok=True)
    cif_names = [x for x in os.listdir(cif_dir) if x.endswith(".cif")]

    # create lammps inputs
    print("Createing CPU pool with " + str(int(os.cpu_count()*0.9)) + " slots...\n\n")
            
    # with multiprocessing.Pool(int(os.cpu_count()*0.9)) as mpool:
    #     mpool.map_async(preprocessing, cif_names).get()

    # run lammps
    Ncpus_per_job = 8
    Npool = int(os.cpu_count() / Ncpus_per_job)

    lmp_jobs = [os.path.join(lmp_dir, x) for x in os.listdir(lmp_dir) if os.path.isdir(os.path.join(lmp_dir, x))]
    input_dicts = [{"Ncpus_per_job": Ncpus_per_job, 
                    "lmp_job_path": lmp_job} for lmp_job in lmp_jobs if "relaxing.210000.restart" not in os.listdir(lmp_job)]
    print("Createing CPU pool with " + str(Npool) + " slots for simulations...\n\n")

    with multiprocessing.Pool(Npool) as mpool:
        mpool.map_async(run_lmp_simulation, input_dicts[:3000]).get()

    
    lmp_jobs = [os.path.join(lmp_dir, x) for x in os.listdir(lmp_dir) if os.path.isdir(os.path.join(lmp_dir, x))]
    input_dicts = [{"Ncpus_per_job": Ncpus_per_job, 
                    "lmp_job_path": lmp_job} for lmp_job in lmp_jobs if "relaxing.210000.restart" in os.listdir(lmp_job)]
    
    Npool = int(os.cpu_count()*0.9)
    print("Createing CPU pool with " + str(Npool) + " slots for analysis...\n\n")

    ret_dicts = None
    with multiprocessing.Pool(Npool) as mpool:
        ret_dicts = mpool.map_async(run_analysis, input_dicts).get()
    mpool.close()
    result = pd.DataFrame(ret_dicts)
    result.to_csv(os.path.join(lmp_dir, "lmp_summary.csv"))

