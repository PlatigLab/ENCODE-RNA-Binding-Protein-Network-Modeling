import logging
import argparse
import wandb
import os

from platiglib.model.model_evaluation import XGBoostModelEvaluation
from platiglib.data.rbpse_dataset import RBPSEDataset                  # here for dynamic class loading
from xgboost import XGBRegressor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration dictionary
param_set = {
    'model': {
        'name': 'XGBRegressor',
        'max_depth': 7,
        'n_estimators': 200,
        'objective': 'reg:logistic',
        'learning_rate': 0.1,
        'reg_alpha': 0.3,
        'subsample': 1,
        'n_jobs': -1,
        "early_stopping_rounds": 100,
        "eval_metric": ["rmse", "logloss"],
    },
    'training': {
        'batch_size': 1024,
        'criterion': 'mean_squared_error',
        'data_split': {
            'method': "set_defs",
            'train_set': ["chr1", "chr3", "chr5", "chr7", "chr9", "chr11", "chr13", "chr15", "chr17", "chr19", "chr21", "chrY"],
            'validate_set': ["chr4", "chr6", "chr10", "chr14", "chr18", "chr22"],
            'test_set': ["chr2", "chr8", "chr12", "chr16", "chr20", "chrX"],
        },
        'scoring': ['r2_score', 'mean_squared_error'],
        'wandb': {
            'track': True,
            'project': "rbp-se-pipeline-dev",  # ignored during sweep
            'entity': 'platiglab',             # ignored during sweep
        }
    },
    'dataset': {
        'name': 'RBPSEDataset',
        'cell_line': 'HepG2',
        'window': 100,
        'binding_format': 'binary',
        # 'exp_norm': 'tmm',
        #'exp_log': False,
        #'df_filter': 'df["RBP_KD"] == "NONE"'
    }
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
    model_evaluation = XGBoostModelEvaluation(model_class, dataset_class, param_set)
    model_evaluation.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run XGBoost model with WandB sweep')
    parser.add_argument('--sweep_id', type=str, required=True, help='WandB sweep ID')
    args = parser.parse_args()

    wandb_param_set = param_set['training']['wandb']
    wandb.agent(args.sweep_id,
                function=main,
                entity=wandb_param_set['entity'],
                project=wandb_param_set['project'],
                count=1)
