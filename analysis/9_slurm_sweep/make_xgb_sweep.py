import os
from functools import reduce
import operator
import subprocess
import wandb
import argparse

# Specify your WandB entity and project
WANDB_ENTITY = "platiglab"
WANDB_PROJECT = "rbp-se-pipeline-dev"

# Define the sweep configuration
sweep_configuration = {
    'method': 'grid',
    'metric': {
        'name': 'test_r2_score',
        'goal': 'maximize'
    },
    'parameters': {
        'model.max_depth': {
            'values': [3, 5, 7, 9],  # Discrete values for max_depth
        },
        'model.reg_alpha': {
            'values': [0.01, 0.03, 0.05],
        }
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
        print(f"Sweep name: {sweep_name}")

    # send enough slurm jobs for the grid search
    num_jobs = calculate_grid_size(sweep_configuration)
    print(f"Sending {num_jobs} slurm jobs...")
    for i in range(num_jobs):
        subprocess.run(
            ['sbatch', '-J', f'xgb_sweep_{sweep_id}', 'run_python_script.slurm', 'wandb_run_xgb.py', '--sweep_id', sweep_id],
            env={k: v for k, v in os.environ.items() if k != 'WANDB_SERVICE'}
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initialize WandB sweep and submit SLURM jobs')
    parser.add_argument('--sweep-name', type=str, help='Optional name for the sweep')
    args = parser.parse_args()

    main(args.sweep_name)