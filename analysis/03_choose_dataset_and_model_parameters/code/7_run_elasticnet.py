import os, glob

SCRIPT_DIR = "/scratch/jve4pt/platiglib/scripts"
YAML_CONFIGS = [os.path.abspath(yaml_file) for yaml_file in glob.glob("../wandb_configs/model/3_linear_model_configs/*.yaml")]
os.chdir(SCRIPT_DIR)

for file in YAML_CONFIGS:

    if "elasticnet" in file.lower(): 
        sweep_name = "ElasticNet"
    elif "ols" in file.lower(): 
        sweep_name = "OLS"

    os.system(
        f'python {SCRIPT_DIR}/send_wandb_sweep.py --memory 128 --cpu 16 --sweep-config {file} --sweep-name {sweep_name}'
    )
