import logging
import argparse
import wandb
import os

import torch
import torch.nn as nn
import numpy as np

from platiglib.model.model_evaluation import PytorchModelEvaluation, get_default_params
from platiglib.data.rbpse_dataset import RBPSEDataset                  # here for dynamic class loading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration dictionary
param_set = {
    'model': {
        'name': 'MLP',
        'hidden_dim': 36,
        'l1_reg': 1e-4,
        'l2_reg': 1e-4,
        'dropout': 0.5,
    },
    'training': {
        'batch_size': 1024,
        'num_epochs': 10,
        'criterion': 'BCELoss',
        'optim': {
            'name': 'Adam',
            'lr': 0.001,
        },
        'data_split': get_default_params('data_split'),
        'profile': get_default_params('profile'),
        'wandb': get_default_params('wandb'),
    },
    'dataset': get_default_params('dataset'),
}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.5, l1_reg=0.1, l2_reg=0.1):
        super(MLP, self).__init__()
        input_dim = np.prod(input_dim)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x.squeeze()

    def regularization(self):
        reg_loss = torch.tensor(0., requires_grad=True)
        for param in self.parameters():
            if self.l1_reg is not None:
                reg_loss = reg_loss + self.l1_reg * torch.norm(param, 1)
            if self.l2_reg is not None:
                reg_loss = reg_loss + self.l2_reg * torch.norm(param, 2)**2
        return reg_loss


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
    model_evaluation = PytorchModelEvaluation(model_class, dataset_class, param_set)
    model_evaluation.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model with WandB sweep')
    parser.add_argument('--sweep_id', type=str, required=True, help='WandB sweep ID')
    args = parser.parse_args()

    wandb_param_set = param_set['training']['wandb']
    wandb.agent(args.sweep_id,
                function=main,
                count=1)
