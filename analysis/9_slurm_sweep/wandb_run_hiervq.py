import logging
import argparse
import tempfile

import wandb
import os

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from platiglib.model.model_evaluation import PytorchModelEvaluation, get_default_params
from platiglib.data.rbpse_dataset import RBPSEDataset                  # here for dynamic class loading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration dictionary
param_set = {
    'model': {
        'name': 'NonNegHierSoftVectorQuantizer',
        'num_codewords': (6, 12),
        'temperature': 0.01,
        'l1_reg': 1e-6,
    },
    'training': {
        'batch_size': 1024,
        'num_epochs': 20,
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


class HierSoftVectorQuantizer(nn.Module):
    def __init__(self, input_dim, num_codewords, temperature=1.0, l1_reg=0.01):
        super(HierSoftVectorQuantizer, self).__init__()
        self.num_rbp = input_dim[0]
        self.num_loc = input_dim[1]
        self.num_codewords = [self.as_vec(num_codewords, 0), self.as_vec(num_codewords, 1)]
        self.temperature = [self.as_vec(temperature, 0), self.as_vec(temperature, 1)]
        self.l1_reg = [self.as_vec(l1_reg, 0), self.as_vec(l1_reg, 1)]

        # First layer codebook
        self.codebook0 = nn.Parameter(torch.randn(self.num_loc, self.num_rbp, self.num_codewords[0]))

        # Second layer codebook
        self.codebook1 = nn.Parameter(torch.randn(self.num_codewords[0] * self.num_loc, self.num_codewords[1]))

        # Output layer
        self.output_layer = nn.Linear(self.num_codewords[1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, num_rbp, num_loc)
        batch_size = x.shape[0]

        # First layer quantization
        similarities0 = torch.einsum('brl,lrc->blc', x, self.codebook0)
        similarities0 = similarities0 / self.temperature[0]
        assignments0 = torch.softmax(similarities0, dim=2)
        combined_assignments0 = assignments0.reshape(batch_size, -1)

        # Second layer quantization
        similarities1 = torch.matmul(combined_assignments0, self.codebook1)
        similarities1 = similarities1 / self.temperature[1]
        assignments1 = torch.softmax(similarities1, dim=1)

        # Final output
        output = self.sigmoid(self.output_layer(assignments1))
        return output.squeeze(-1)

    def regularization(self):
        reg0 = self.l1_reg[0] * torch.norm(self.codebook0, 1)
        reg1 = self.l1_reg[1] * torch.norm(self.codebook1, 1)
        return reg0 + reg1

    def store_wandb_artifacts(self, dataset_info=None):
        self._store_wandb_artifacts(self.codebook0, self.codebook1, dataset_info)

    def _store_wandb_artifacts(self, codebook0, codebook1, dataset_info=None):
        codebook0 = codebook0.detach().cpu().numpy()
        codebook1 = codebook1.detach().cpu().numpy()
        codebook1 = codebook1.reshape((self.num_loc, self.num_codewords[0], self.num_codewords[1]))

        if dataset_info is not None:
            rbp_index = dataset_info['rbp_index']
            loc_index = dataset_info['loc_index']
        else:
            rbp_index = pd.Index(range(self.num_rbp))
            loc_index = pd.Index(range(self.num_loc))

        def save_codebooks_artifact(df, cb_idx):
            all_dfs = []
            for i in range(self.num_codewords[cb_idx]):
                new_df = df.xs(i, level=f'codebook{cb_idx}_idx').unstack(level='loc')
                new_df.columns = loc_index
                new_df[f'codebook{cb_idx}_idx'] = i
                all_dfs.append(new_df)

            save_df = pd.concat(all_dfs, ignore_index=False)
            cols_to_save = [f'codebook{cb_idx}_idx'] + loc_index.tolist()
            save_df_final = save_df[cols_to_save].reset_index()
            cols = list(save_df_final.columns)
            cols[0], cols[1] = cols[1], cols[0]
            save_df_final = save_df_final[cols]

            with tempfile.TemporaryDirectory() as temp_dir:
                csv_path = os.path.join(temp_dir, f"codebook{cb_idx}.csv")
                save_df_final.to_csv(csv_path, sep='\t', index=False)

                artifact = wandb.Artifact(name=f"codebook{cb_idx}-{wandb.run.name}", type="dataset",
                                          description=f"softVQ model codebook{cb_idx}")
                artifact.add_file(csv_path)
                wandb.log_artifact(artifact)

        # save codebook0
        iterables = [loc_index, rbp_index, range(self.num_codewords[0])]
        multi_index = pd.MultiIndex.from_product(
            iterables,
            names=['loc', 'rbp', 'codebook0_idx']
        )
        df = pd.DataFrame({'value': codebook0.flatten()}, index=multi_index)
        save_codebooks_artifact(df, 0)

        # save codebook1
        iterables = [loc_index, range(self.num_codewords[0]), range(self.num_codewords[1])]
        multi_index = pd.MultiIndex.from_product(
            iterables,
            names=['loc', 'codebook0_idx', 'codebook1_idx']
        )
        df = pd.DataFrame({'value': codebook1.flatten()}, index=multi_index)
        save_codebooks_artifact(df, 1)



    @staticmethod
    def as_vec(var, d):
        if isinstance(var, (int, float)):
            return var
        if isinstance(var, (list, tuple, np.ndarray)):
            if len(var) == 1:
                return var
            else:
                return var[d]
        else:
            raise ValueError(f"bad type for var {type(var)}")


class NonNegHierSoftVectorQuantizer(HierSoftVectorQuantizer):
    def forward(self, x):
        # x shape: (batch_size, num_rbp, num_loc)
        batch_size = x.shape[0]

        # First layer quantization
        nonneg_codebook0 = torch.softmax(self.codebook0 / self.temperature[0], dim=1)
        assignments0 = torch.einsum('brl,lrc->blc', x, nonneg_codebook0)
        combined_assignments0 = assignments0.reshape(batch_size, -1)

        # Second layer quantization
        nonneg_codebook1 = torch.softmax(self.codebook1 / self.temperature[1], dim=0)
        assignments1 = torch.matmul(combined_assignments0, nonneg_codebook1)

        # Final output
        output = self.sigmoid(self.output_layer(assignments1))
        return output.squeeze(-1)

    def store_wandb_artifacts(self, dataset_info=None):
        nonneg_codebook0 = torch.softmax(self.codebook0 / self.temperature[0], dim=1)
        nonneg_codebook1 = torch.softmax(self.codebook1 / self.temperature[1], dim=0)
        self._store_wandb_artifacts(nonneg_codebook0, nonneg_codebook1, dataset_info)



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
