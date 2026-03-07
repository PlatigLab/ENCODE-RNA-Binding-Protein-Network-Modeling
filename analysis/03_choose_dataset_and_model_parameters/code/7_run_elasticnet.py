import os, glob

SCRIPT_DIR = "/scratch/jve4pt/platiglib/scripts"
YAML_CONFIG = os.path.abspath(
    "../wandb_configs/model/3_linear_model_configs/1_elasticnet.yaml"
)

os.chdir(SCRIPT_DIR)


os.system(
    f'python3.11 {SCRIPT_DIR}/send_wandb_sweep.py '
    '--memory 128 '
    '--cpu 16 '
    f'--sweep-config {YAML_CONFIG} '
    '--time=08:00:00 '
    '--account=platiglab_paid '
    '--max-agents 100'
)
