import os, glob

SCRIPT_DIR = "/scratch/jve4pt/platiglib/scripts"
ONE_DIM_CONFIG_DIR = "/project/PlatigLab/users/yogi/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/03_choose_dataset_and_model_parameters/wandb_configs/model/1D-sweeps/"

os.chdir(SCRIPT_DIR)

config_files = glob.glob(os.path.join(ONE_DIM_CONFIG_DIR, "*.yaml"))
for config_file in config_files:

    os.system(
        f'python {SCRIPT_DIR}/send_wandb_sweep.py --memory 256 --cpu 16 --sweep-config {config_file} --sweep-name 1D_{config_file.split("/")[-1].split(".yaml")[0]}'
    )
