import os

CELL_LINES = ["HepG2", "K562"]
SCRIPT_DIR = "/scratch/jve4pt/platiglib/scripts"
YAML_FILE_PREFIX= "../wandb_configs/model/2_sequential_sweeps/4_regularization"

# Get the cell line yaml files
cell_line_yaml_files = [
        os.path.abspath(
            f"{YAML_FILE_PREFIX}_{cell_line}.yaml"
        )
        for cell_line in CELL_LINES
    ]

os.chdir(SCRIPT_DIR)

for yaml_file in cell_line_yaml_files:
    # Run the sweep
    os.system(
        f'python3.11 {SCRIPT_DIR}/send_wandb_sweep.py '
        '--memory 128 '
        '--cpu 16 '
        f'--sweep-config {yaml_file} '
        '--time=08:00:00 '
        '--account=platiglab_paid '
        '--max-agents 100'    
    )
