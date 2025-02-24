import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, polars as pl
import wandb, json, pickle

from dataclasses import dataclass
from loguru import logger
from pathlib import Path

@dataclass
class DatasetAndModelParameterAnalyzer:

    sweep_projects = {
        'dataset': 'yogi-dataset-sweep-feb-2025', 
    }
    dataset_sweep_covariates= {
        'dataset.min_read_count': "Min. Read Count", 
        'dataset.features.binding_matrix.binding_format': "eCLIP Data Type", 
        'dataset.features.binding_matrix.window': "Junction Window"
    }

    def __post_init__(self):

        self.retrieve_wandb_summary_tables()
        self.run_summary_table_assertions()
    

    def retrieve_wandb_summary_tables(self):
        
        self.sweep_results = {}
    
        output_folder = Path("../output/wandb_summary_tables/")
        if len(list(output_folder.glob('*'))) == 1: 
            
            self.sweep_results = {file.stem.split("_")[0]: pd.read_csv(file, sep="\t") for file in output_folder.glob('*')}
            logger.info("FROM CACHE: Retrieved WandB summary tables")

        else: 
            
            logger.info("Retrieving summary tables from WandB")

            for type in self.sweep_projects: 
                runs = wandb.Api().runs(self.sweep_projects[type])

                run_data = []
                for run in runs:

                    run_info = {
                        "run_id": run.id,
                        "run_name": run.name,
                        "state": run.state,
                        "sweep_name": run.sweep.name if run.sweep else None,
                        "sweep_id": run.sweep.id if run.sweep else None
                    }

                    # Extract config values
                    config = json.loads(run.json_config)
                    for key, item in config.items():
                        if key != "_wandb":  # Skip wandb metadata
                            run_info[key] = item["value"]

                    # Add metrics
                    metrics = {k: v for k, v in run.summary._json_dict.items() if isinstance(v, float)}
                    run_info.update(metrics)

                    run_data.append(run_info)

                run_df = pd.DataFrame(run_data)
                run_df.to_csv(output_folder / f"{type}_sweep_summary.tsv", index=False, sep="\t")
                self.sweep_results[type] = run_df
            
            logger.success("Retrieved & cached WandB summary tables")


    def run_summary_table_assertions(self):

        logger.info("Running assertions on summary tables")

        for key in self.sweep_results:
            table = self.sweep_results[key]

            assert table['run_id'].is_unique, "run_id column contains duplicated values"
            assert table['run_name'].is_unique, "run_name column contains duplicated values"
            assert (table['state'] == 'finished').all(), "Not all runs are finished"
            assert table['sweep_name'].nunique() == 1, "sweep_name column contains multiple unique values"
            assert table['sweep_id'].nunique() == 1, "sweep_id column contains multiple unique values"

            if key=='dataset': 
                assert table[
                    [col for col in table.columns if col.startswith("dataset.")]
                ].duplicated().sum() == 0, "There are duplicate rows in the dataset columns"

        logger.success("All assertions passed")


    def calculate_num_examples_and_average_binding(self): 

        pkl_file = Path("../output/data_matrix_stats/").glob("*.pkl")

        if len(pkl_file) == 1:
            logger.info("FROM CACHE: Loading num examples and average binding from pickle file")

            with open(pkl_file[0], 'rb') as f:
                data_matrix_stats = pickle.load(f)

        else:
            logger.info("Calculating num examples and average binding")

            data_matrix_stats = {}

            for cell_line in self.sweep_results['dataset']['dataset.cell_line'].unique(): 
                for window in self.sweep_results['dataset']['dataset.features.binding_matrix.window'].unique():

                    tmp_df = pl.scan_csv(
                        f"/project/PlatigLab/data/RBP_ML/3_yogi_dataset_feb_2025/{cell_line}_{window}_all-events_num-peaks-no-kd.tsv.gz", 
                        has_header=True, 
                        separator='\t'
                    )

                    


                    

    
    def plot_r2_distributions(self): 

        for type in self.sweep_results:

            data = self.sweep_results[type]

            plt.figure(figsize=(5,3), dpi=200)

            sns.swarmplot(x='dataset.cell_line', y='val_r2_score', data=data, palette=['lightblue', 'lightcoral'], linewidth=1, edgecolor='black')
            sns.boxplot(x='dataset.cell_line', y='val_r2_score', data=data, showcaps=False, boxprops={'facecolor':'None', 'edgecolor':'black'}, whiskerprops={'color':'black', 'linewidth':2}, medianprops={'color':'black'}, showfliers=False)

            plt.title(f'{type.capitalize()} Sweep: R2 Score Distribution')
            plt.xlabel('Cell Line')
            plt.ylabel('Test R2 Score')
            plt.show()
    
    
    def plot_r2_per_covariate(self, type=None): 

        assert type in self.sweep_results, f"Type {type} not found in sweep results"

        if type == 'dataset': 
            covariates = self.dataset_sweep_covariates
        
        for covariate in covariates:
            plt.figure(figsize=(8, 3), dpi=300)

            swarm = sns.swarmplot(x='dataset.cell_line', y='val_r2_score', hue=covariate, data=self.sweep_results[type], dodge=True, palette='Set2', linewidth=1, edgecolor='black')
            sns.boxplot(x='dataset.cell_line', y='val_r2_score', hue=covariate, data=self.sweep_results[type], dodge=True, showcaps=False, boxprops={'facecolor':'None', 'edgecolor':'black'}, whiskerprops={'color':'black', 'linewidth':2}, medianprops={'color':'black'}, showfliers=False)

            # Remove the boxplot legend and place it outside the plot in the middle
            handles, labels = swarm.get_legend_handles_labels()
            plt.legend(handles=handles, labels=labels, title=covariate.split(".")[-1], bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0.)

            plt.title(f'{type.capitalize()} Sweep: R2 Score by {covariate.split(".")[-1]}')
            plt.xlabel('Cell Line')
            plt.ylabel('Test R2 Score')

            plt.show()
            plt.close()
