import os

SCRIPT_DIR = "/scratch/jve4pt/platiglib/scripts"
YAML_FILE = "../wandb_configs/model/2_sequential_sweeps/1_trees_and_learning.yaml"

YAML_ABS_PATH = os.path.abspath(YAML_FILE)
os.chdir(SCRIPT_DIR)

os.system(
    f'python3.11 {SCRIPT_DIR}/send_wandb_sweep.py '
    '--memory 128 '
    '--cpu 16 '
    f'--sweep-config {YAML_ABS_PATH} '
    '--time=08:00:00 '
    '--account=platiglab_paid '
    '--max-agents 200'
)
