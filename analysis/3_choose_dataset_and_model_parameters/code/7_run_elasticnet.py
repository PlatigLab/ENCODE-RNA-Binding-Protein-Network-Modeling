import os

SCRIPT_DIR = "/scratch/jve4pt/platiglib/scripts"
YAML_FILE = "../wandb_configs/model/3_linear_model_configs/1_elasticnet.yaml"
YAML_ABS_PATH = os.path.abspath(YAML_FILE)
os.chdir(SCRIPT_DIR)

os.system(
    f'python {SCRIPT_DIR}/send_wandb_sweep.py --memory 128 --cpu 16 --sweep-config {YAML_ABS_PATH} --sweep-name "ElasticNet" --max-agents 150 --time 12:00:00'
)
