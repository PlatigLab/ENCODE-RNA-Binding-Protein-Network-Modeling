import os
from functools import reduce
import operator
import subprocess
import wandb
import argparse

# WandB entity and project
WANDB_ENTITY = "platiglab"
WANDB_PROJECT = "rbp-se-pipeline-dev2"

MODEL_NAME = 'hiervq'
SCRIPT_NAME = f'wandb_run_{MODEL_NAME}.py'
SWEEP_NAME = f'{MODEL_NAME}_sweep'
MEMORY_GB = 64
CPUS = 4
USE_GPU = True

sweep_configuration = {
    'method': 'grid',
    'metric': {
        'name': 'val_r2_score',
        'goal': 'maximize'
    },
    'parameters': {
        # 'training.optim.lr': {
        #     'values': [1e-5, 1e-3, 1e-1],
        # },
        # 'model.ent_reg': {
        #     'values': [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        # },
        # 'model.allow_negative': {
        #     'values': [False, True],
        # },
        'model.l2_reg': {
           'values': [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        },
        'model.temperature': {
           # 'values': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0],
            'values': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
        },
        #'dataset.cell_line': {
         #    'values': ['HepG2', 'K562'],
         #},
        #'dataset.min_read_count': {
            #'values': [None, 10, 20, 40, 100, 200, 400],
        #},
        #'dataset.window': {
            #'values': [50, 75, 100, 125, 150, 175, 200, 225, 250],
        #},
        #'dataset.binding_format': {
            #'values': ['peak_count', 'binary', 'rbp_exp', 'rbp_exp_peak'],
        #},
        # 'model.n_estimators': {
        #     'values': [100, 200, 500],
        # },
        # 'model.nhead': {
        #     'values': [1, 2, 4],
        # },
        # 'model.d_model': {
        #     'values': [4, 8, 16, 32, 64],
        # },
        # 'model.dropout': {
        #     'values': [0.1, 0.5],
        # },
        # 'model.num_encoder_layers': {
        #     'values': [1, 2],
        # },
        # 'model.dim_feedforward': {
        #     'values': [8, 16, 64, 128],
        # },
        #  'training.seed': {
        #      'values': list(range(1000, 1100)),#[4232, 451, 2352],#, 321, 9491],
        #  },
        # 'model.hidden_dim': {
        #     'values': [4, 6, 8, 12, 16, 20, 40],
        # },
        # 'model.num_codewords': {
        #     'values': [4, 6, 8, 12],
        # },
        # 'model.num_codewords': {
        #     'values': [(i, j) for i in [4, 6] for j in [6, 8, 10, 12]],
        # },
        #'model.alpha': {
        #   'values': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0],
        #},
        # 'model.comm_reg': {
        #    'values': [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        # },
        # 'model.comm_beta': {
        #     'values': [0.1, 0.25, 0.5, 0.75, 0.9],
        # },
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


def main(sweep_id=None, sweep_name=None):
    # Initialize the sweep
    if sweep_id is None:
        # Add sweep name to configuration if provided
        if sweep_name is None:
            sweep_name = SWEEP_NAME
        print(f"Sweep name: {sweep_name}\n")
        sweep_configuration['name'] = sweep_name

        sweep_id = wandb.sweep(sweep_configuration,
                               project=WANDB_PROJECT,
                               entity=WANDB_ENTITY)

        print(f"New sweep initialized.")
    print(f"Sweep ID: {sweep_id}")


    if USE_GPU:
        slurm_script_name = 'run_python_script_gpu.slurm'
    else:
        slurm_script_name = 'run_python_script.slurm'

    # send enough slurm jobs for the grid search
    num_jobs = calculate_grid_size(sweep_configuration)
    print(f"Sending {num_jobs} slurm jobs...")
    for i in range(num_jobs):
        subprocess.run(['sbatch',
                        f'--job-name={sweep_name}_{sweep_id}_{i:02}',
                        f'--mem={MEMORY_GB}G',
                        f'--cpus-per-task={CPUS}',
                        '--exclude=udc-an38-13',  # manual hpc node blacklist
                        slurm_script_name,
                        SCRIPT_NAME,
                        '--sweep_id', sweep_id],
            env={k: v for k, v in os.environ.items() if k != 'WANDB_SERVICE'}
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initialize WandB sweep and submit SLURM jobs')
    parser.add_argument('--sweep-id', type=str, help='Optional id of  existing sweep')
    parser.add_argument('--sweep-name', type=str, help='Optional name for the sweep')
    args = parser.parse_args()

    main(sweep_id=args.sweep_id, sweep_name=args.sweep_name)
