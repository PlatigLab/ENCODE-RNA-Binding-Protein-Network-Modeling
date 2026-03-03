import os, glob

SCRIPT_DIR = "/scratch/jve4pt/platiglib/scripts"
DATA_CACHE_YAML = "../wandb_configs/create_data_cache.yaml"

config_file = os.path.abspath(DATA_CACHE_YAML)
os.chdir(SCRIPT_DIR)

os.system(
    f'python3.11 {SCRIPT_DIR}/send_wandb_sweep.py \
        --memory 100 \
        --cpus 20 \
        --account=platiglab_paid \
        --sweep-config {config_file}'
)