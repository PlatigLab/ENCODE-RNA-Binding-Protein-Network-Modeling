import logging
import argparse
import wandb
import os

from platiglib.model.model_evaluation import SklearnModelEvaluation, get_default_params
from platiglib.data.rbpse_dataset import RBPSEDataset                  # here for dynamic class loading
from sklearn.linear_model import ElasticNet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration dictionary
param_set = {
    'model': {
        'name': 'ElasticNet',
        'l1_ratio': 0.5,
        'alpha': 1.0,
        'random_state': 42,
    },
    'training': {
        'batch_size': 1024,
        'criterion': 'mean_squared_error',
        'data_split': get_default_params('data_split'),
        'profile': get_default_params('profile'),
        'wandb': get_default_params('wandb'),
    },
    'dataset': get_default_params('dataset'),
}


def update_nested_dict(d, key, value):
    """Update a nested dictionary with a dot-separated key."""
    keys = key.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def main():
    # Initialize wandb
    with wandb.init() as run:
        # Update param_set with values from wandb.config
        for key, value in wandb.config.items():
            update_nested_dict(param_set, key, value)

        # Log the updated param_set
        wandb.config.update(param_set)

    # print(param_set['model'])
    model_class = globals()[param_set['model']['name']]
    dataset_class = globals()[param_set['dataset']['name']]
    model_evaluation = SklearnModelEvaluation(model_class, dataset_class, param_set)
    model_evaluation.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Sklearn model with WandB sweep')
    parser.add_argument('--sweep_id', type=str, required=True, help='WandB sweep ID')
    args = parser.parse_args()

    wandb_param_set = param_set['training']['wandb']
    wandb.agent(args.sweep_id,
                function=main,
                count=1)
