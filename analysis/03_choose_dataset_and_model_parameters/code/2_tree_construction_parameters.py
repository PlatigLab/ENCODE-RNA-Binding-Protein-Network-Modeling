import os

SCRIPT_DIR = "/scratch/jve4pt/platiglib/scripts"
YAML_FILE = "../wandb_configs/model/2_sequential_sweeps/1_trees_and_learning.yaml"

YAML_ABS_PATH = os.path.abspath(YAML_FILE)
os.chdir(SCRIPT_DIR)

os.system(
    f'python {SCRIPT_DIR}/send_wandb_sweep.py --memory 128 --cpu 16 --sweep-config {YAML_ABS_PATH} --sweep-name "1:early_stopping_rounds-learning_rate-max_depth" --max-agents 500'
)
