import logging
import argparse
import wandb
import os

import torch
import torch.nn as nn
import numpy as np

from platiglib.model.evaluation import PytorchModelEvaluation, get_default_params
from platiglib.data.rbpse_dataset import RBPSEDataset                  # here for dynamic class loading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration dictionary
param_set = {
    'model': {
        'name': 'SoftVectorQuantizer',
        'num_codewords': 4,
        'temperature': 0.25,
        'l1_reg': 1e-3,
    },
    'training': {
        'batch_size': 1024,
        'num_epochs': 4,
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


class SoftVectorQuantizer(nn.Module):
    def __init__(self, input_dim, num_codewords, temperature=1.0, l1_reg=0.01):
        super(SoftVectorQuantizer, self).__init__()
        self.num_rbp = input_dim[0]
        self.num_codewords = num_codewords
        self.num_loc = input_dim[1]
        self.temperature = temperature
        self.l1_reg = l1_reg

        # Create a single codebook tensor holding all locations
        # Shape: (num_loc, num_rbp, num_codewords)
        self.codebooks = nn.Parameter(torch.randn(self.num_loc, self.num_rbp, self.num_codewords))

        # Output layer takes all assignments as input
        self.output_layer = nn.Linear(num_codewords * self.num_loc, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, num_rbp, num_loc)
        batch_size = x.shape[0]

        # Compute dot products for all locations at once
        # Each location has num_codewords code vectors that are considered in turn
        # x: (batch_size, num_rbp, num_loc)
        # self.codebooks: (num_loc, num_rbp, num_codewords)
        # similarities: (batch_size, num_loc, num_codewords)
        similarities = torch.einsum('brl,lrc->blc', x, self.codebooks)
        similarities = similarities / self.temperature

        # Compute soft assignments for all locations (so sum along codewords dim = 1)
        assignments = torch.softmax(similarities, dim=2)

        # Reshape assignments to (batch_size, num_loc * num_codewords)
        combined_assignments = assignments.reshape(batch_size, -1)

        # Compute output and apply sigmoid
        output = self.sigmoid(self.output_layer(combined_assignments))

        return output.squeeze(-1)

    def regularization(self):
        return self.l1_reg * torch.norm(self.codebooks, 1)


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
