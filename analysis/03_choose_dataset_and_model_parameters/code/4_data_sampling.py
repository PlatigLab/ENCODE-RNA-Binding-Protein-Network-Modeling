import os

CELL_LINES = ["K562", "HepG2"]
SCRIPT_DIR = "/scratch/jve4pt/platiglib/scripts"
YAML_FILE_PREFIX= "/project/PlatigLab/users/yogi/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/03_choose_dataset_and_model_parameters/wandb_configs/model/2_sequential_sweeps/3_data_sampling"

os.chdir(SCRIPT_DIR)

for cell_line in CELL_LINES:
    YAML_FILE = f"{YAML_FILE_PREFIX}_{cell_line}.yaml"
    # Run the sweep
    os.system(
        f'python {SCRIPT_DIR}/send_wandb_sweep.py --memory 128 --cpu 16 --sweep-config {YAML_FILE} --sweep-name "3:subsample-colsample_bytree:{cell_line}"'
    )
