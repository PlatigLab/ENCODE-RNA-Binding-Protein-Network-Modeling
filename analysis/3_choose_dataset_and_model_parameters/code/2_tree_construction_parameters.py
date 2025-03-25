import os

SCRIPT_DIR = "/scratch/jve4pt/platiglib/scripts"
YAML_FILE = "/project/PlatigLab/users/yogi/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/4_choose_dataset_and_model_parameters/wandb_configs/model/sequential_sweeps/1_trees_and_learning.yaml"

os.chdir(SCRIPT_DIR)
os.system(
    f'python {SCRIPT_DIR}/send_wandb_sweep.py --memory 256 --cpu 16 --sweep-config {YAML_FILE} --sweep-name "1:n_estimators-learning_rate-max_depth" --max-agents 1000'
)
