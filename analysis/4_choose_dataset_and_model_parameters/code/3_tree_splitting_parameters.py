import os

SCRIPT_DIR = "/scratch/jve4pt/platiglib/scripts"
YAML_FILE = "/project/PlatigLab/users/yogi/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/4_choose_dataset_and_model_parameters/wandb_configs/model/sequential_sweeps/2_tree_splitting_parameters.yaml"

os.chdir(SCRIPT_DIR)
os.system(
    f'python {SCRIPT_DIR}/send_wandb_sweep.py --memory 128 --cpu 16 --sweep-config {YAML_FILE} --sweep-name "2:min_child_weight-gamma"'
)
