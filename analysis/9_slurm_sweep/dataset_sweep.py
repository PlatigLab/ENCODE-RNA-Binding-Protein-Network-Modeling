import os
from functools import reduce
import operator
import subprocess
import wandb
import argparse

# WandB entity and project
WANDB_ENTITY = "platiglab"
WANDB_PROJECT = "rbp-se"

SCRIPT_NAME = 'wandb_run_xgb.py'
SLURM_JOB_NAME = 'xgb_sweep'
MEMORY_GB = 128
CPUS = 8
USE_GPU = False

sweep_configuration = {
    'method': 'grid',
    'metric': {
        'name': 'val_r2_score',
        'goal': 'maximize'
    },
    'parameters': {
        # 'dataset.cell_line': {
        #     'values': ['HepG2', 'K562'],
        # },
        'dataset.window': {
            'values': [50, 75, 100, 125, 150, 175, 200, 225, 250],
        },
        'dataset.binding_format': {
            'values': ['peak_count', 'binary', 'rbp_exp', 'rbp_exp_peak'],
        },
        'model.n_estimators': {
            'values': [100, 200, 500],
        },
    }
}


def calculate_grid_size(sweep_config):
    parameters = sweep_config['parameters']
    grid_dimensions = []
    for param, config in parameters.items():
        if 'values' in config:
            grid_dimensions.append(len(config['values']))
        elif all(k in config for k in ('min', 'max', 'step')):
            num_values = int((config['max'] - config['min']) / config['step']) + 1
            grid_dimensions.append(num_values)
        else:
            print(f"Warning: Parameter {param} is not discretized. Assuming 1 value.")
            grid_dimensions.append(1)
    total_combinations = reduce(operator.mul, grid_dimensions, 1)
    return total_combinations


def main(sweep_name=None):
    # Add sweep name to configuration if provided
    if sweep_name:
        sweep_configuration['name'] = sweep_name

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_configuration,
                           project=WANDB_PROJECT,
                           entity=WANDB_ENTITY)

    print(f"Sweep initialized. Sweep ID: {sweep_id}")
    if sweep_name:
        print(f"Sweep name: {sweep_name}\n")

    if USE_GPU:
        slurm_script_name = 'run_python_script_gpu.slurm'
    else:
        slurm_script_name = 'run_python_script.slurm'

    # send enough slurm jobs for the grid search
    num_jobs = calculate_grid_size(sweep_configuration)
    print(f"Sending {num_jobs} slurm jobs...")
    for i in range(num_jobs):
        subprocess.run(['sbatch',
                        f'--job-name={SLURM_JOB_NAME}_{sweep_id}_{i:02}',
                        f'--mem={MEMORY_GB}G',
                        f'--cpus-per-task={CPUS}',
                        slurm_script_name,
                        SCRIPT_NAME,
                        '--sweep_id', sweep_id],
            env={k: v for k, v in os.environ.items() if k != 'WANDB_SERVICE'}
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initialize WandB sweep and submit SLURM jobs')
    parser.add_argument('--sweep-name', type=str, help='Optional name for the sweep')
    args = parser.parse_args()

    main(args.sweep_name)
