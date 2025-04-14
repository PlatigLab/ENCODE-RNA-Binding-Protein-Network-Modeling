import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, polars as pl, numpy as np
import wandb, json, argparse, os, sys, glob

from dataclasses import dataclass
from loguru import logger
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

@dataclass
class DatasetAndModelParameterAnalyzer:

    sweep_projects = {
        'dataset': 'yogi-dataset-sweep-feb-2025', 
        'model': 'yogi-xgbregressor-hyperparameter-sweep-april-2025',
        'linear_models': 'yogi-RBP-ML-linear-models-april-2025'
    }

    dataset_sweep_covariates= {
        'dataset.min_read_count': "Min. Read Count", 
        'dataset.features.binding_matrix.binding_format': "Binding Data Format", 
        'dataset.features.binding_matrix.window': "Window"
    }

    view_n_configs = 10
    choose_n_configs = 5

    def __post_init__(self):

        self.retrieve_wandb_summary_tables()


    def retrieve_wandb_summary_tables(self):
        
        self.sweep_results = {}
    
        output_folder = Path("../output/wandb_summary_tables/")
        if len(list(output_folder.glob('*'))) == 3: 
            
            self.sweep_results = {file.stem.split("_")[0]: pd.read_csv(file, sep="\t") for file in output_folder.glob('*')}

            logger.warning("REMINDER: removing early_stopping_rounds outside of 100")
            model_df = self.sweep_results['model']
            model_df = model_df[model_df['model.early_stopping_rounds'] == 100]
            self.sweep_results['model'] = model_df

            for key, df in self.sweep_results.items():
                logger.info(f"{key} summary table contains {df.shape[0]} rows")

            logger.success("FROM CACHE: Retrieved WandB summary tables")

        else: 
            
            logger.info("Retrieving summary tables from WandB")

            for type in self.sweep_projects: 
                output_file = output_folder / f"{type}_sweep_summary.tsv"
                if not output_file.exists():
                    logger.info(f"Retrieving {type} sweep summary table from WandB")
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

                    if type == 'model':
                        for row in run_data:
                            if row["sweep_name"].count(":") == 2:
                                row["sweep_name"] = row["sweep_name"].rsplit(":", 1)[0]
                                
                    run_df = pd.DataFrame(run_data)
                    run_df.to_csv(output_folder / f"{type}_sweep_summary.tsv", index=False, sep="\t")
                    self.sweep_results[type] = run_df
            
            logger.success("Retrieved & cached WandB summary tables")


    def run_summary_table_assertions(self):

        logger.info("Running assertions on summary tables")

        for key in self.sweep_results:
            logger.info(f"Running assertions for {key} summary table")
            table = self.sweep_results[key].copy(deep=True)

            assert table['run_id'].is_unique, "run_id column contains duplicated values"
            assert (table['state'] == 'finished').all(), "Not all runs are finished"

            for col in table.columns:
                if table[col].apply(lambda x: isinstance(x, (list, dict, set))).any():
                    table.drop(columns=[col], inplace=True)
                        
            if key =='dataset': 
                assert table['sweep_name'].nunique() == 1, "sweep_name column contains multiple unique values"
                assert table['sweep_id'].nunique() == 1, "sweep_id column contains multiple unique values"

                dataset_cols = [col for col in table.columns if col.startswith("dataset.")]
                assert table[dataset_cols].duplicated().sum() == 0, "There are duplicate rows in the dataset columns"

            elif key == 'model':
                assert table['holdout_r2_score'].apply(lambda x: isinstance(x, float) and not pd.isna(x)).all(), "Not all values in holdout_r2_score are valid decimal float values"
                assert (table['training.seed'] != 0).all() and (table['training.seed'] != 0.0).all(), "Some runs have training.seed equal to 0"
                
                group_cols = [col for col in table.columns if col.startswith("dataset.") or col.startswith("model.") or col.startswith("training.")]

                for sweep_id in table['sweep_id'].unique():
                    subset_table = table[table['sweep_id'] == sweep_id]
                    assert subset_table[group_cols].duplicated().sum() == 0, f"There are duplicate rows in the group columns for sweep_id {sweep_id}"

        logger.success("All assertions passed")


    def calculate_num_examples_and_average_binding(self, cell_line, window, min_read_count): 
        min_read_count = int(min_read_count)
        
        df = pl.scan_csv(
            f"/project/PlatigLab/data/RBP_ML/3_yogi_dataset_feb_2025/{cell_line}_{window}_all-events_num-peaks-no-kd.tsv.gz", 
            has_header=True, 
            separator='\t'
        ).filter(
            pl.col("Total Read Counts") > min_read_count
        ).collect()
        assert df['index'].n_unique() == df.shape[0], "The 'index' column contains duplicate values"

        validate_count = df.filter(pl.col("chr").is_in(self.validate_set)).shape[0]
        test_count = df.filter(pl.col("chr").is_in(self.test_set)).shape[0]

        binding_cols = [col for col in df.columns if col.endswith("_binding")]
        df = df.with_columns([
            pl.when(pl.col(col) > 1).then(1).otherwise(pl.col(col)).alias(col) 
            for col in binding_cols
        ])
        
        assert all(df[col].max() <= 1 for col in binding_cols), "Some values in binding columns are greater than 1"
        original_binding_data_shape = df.shape

        unique_rbp_kd_targets = sorted(df["RBP_KD_Target"].unique().to_list())
        modified_dfs = []
        for rbp_kd_target in unique_rbp_kd_targets:
            subset_df = df.filter(pl.col("RBP_KD_Target") == rbp_kd_target)

            if rbp_kd_target != "CTRL":
                binding_cols_to_zero = [col for col in binding_cols if col.startswith(f"{rbp_kd_target}_")]
                assert len(binding_cols_to_zero) ==6, print(binding_cols_to_zero)
                
                subset_df = subset_df.with_columns([
                    pl.lit(0).alias(col) for col in binding_cols_to_zero
                ])

            modified_dfs.append(subset_df)

        df = pl.concat(modified_dfs, how='vertical_relaxed')
        assert df.shape == original_binding_data_shape, "Dataframe shape changed after modification"

        horizontal_sum = df.select(binding_cols).sum_horizontal()
        average_binding = (horizontal_sum.sum()) / (df.shape[0] * 6)

        output_dict = {
            "cell_line": cell_line,
            "window": window,
            "min_read_count": min_read_count,
            "avg_binding_per_graph_per_window": average_binding,
            "total_examples": df.shape[0],
            "validate_count": validate_count,
            "test_count": test_count
        }

        output_file = f"../output/data_matrix_stats/{cell_line}_{window}_{min_read_count}_stats.json"
        with open(output_file, 'w') as f:
            json.dump(output_dict, f, indent=4)
        
        # Plot the histogram of the distribution of these sums
        plt.figure(figsize=(7, 4), dpi=200)

        bins = list(range(1, 12))  # Bins from 1 to 10, and one bin for >10
        horizontal_sum = horizontal_sum.clip(upper_bound=11)  # Clip values greater than 10 to 11
        sns.histplot(horizontal_sum, bins=bins, color='skyblue', edgecolor='black')

        plt.title(f'{cell_line}, Window: {window}, Min Read Count: {min_read_count}\nValues greater than 10 clipped to 11', y=1.01)
        plt.xlabel('# Bindings per Graph')
        plt.ylabel('Frequency')

        # Save the plot
        plot_file = f"../output/data_matrix_stats/{cell_line}_{window}_{min_read_count}_bindings_per_graph.png"
        plt.savefig(plot_file, dpi=200)
        plt.close()

        # Plot the histogram of the "Target_PSI" values
        plt.figure(figsize=(7, 4), dpi=200)
        sns.histplot(df['Target_PSI'], bins=50, color='skyblue', edgecolor='black')

        plt.title(f'{cell_line}, Window: {window}, Min Read Count: {min_read_count}\nDistribution of Target_PSI', y=1.01)
        plt.xlabel('Target_PSI')
        plt.ylabel('Frequency')

        # Save the plot
        psi_plot_file = f"../output/data_matrix_stats/{cell_line}_{window}_{min_read_count}_target_psi_distribution.png"
        plt.savefig(psi_plot_file, dpi=200)
        plt.close()

        logger.success(f"Calculated and saved stats for {cell_line}, {window}, {min_read_count}")


    def average_across_seed_per_cell_line(self, data= None, group_by=None):
        assert group_by is not None and data is not None, "group_by and data should be provided"

        grouped = data.groupby(group_by + ['dataset.cell_line'])
        assert all(len(group) == 9 for _, group in grouped), "Not all groups have exactly 9 entries"
        
        average_per_config = grouped['holdout_r2_score'].mean().reset_index().rename(columns={'holdout_r2_score': 'avg_holdout_r2_score'})
        
        combined_configs = []
        for cell_line in average_per_config['dataset.cell_line'].unique().tolist():

            top_configs_cell_line = average_per_config[
                    average_per_config['dataset.cell_line'] == cell_line
                ].sort_values(
                    by='avg_holdout_r2_score',
                    ascending=False
                ).head(self.view_n_configs)
            
            combined_configs.append(top_configs_cell_line)

        return pd.concat(combined_configs, ignore_index=True)


    def show_top_model_configs_after_averaging_by_seed(self, all_configs=None, linear=False): 

        assert all_configs is not None, "all_configs should be provided"

        if linear: 
            model_sweep = self.sweep_results['linear'].copy(deep=True)
        elif not linear: 
            model_sweep = self.sweep_results['model'].copy(deep=True)
            model_sweep = model_sweep[
                    model_sweep['sweep_name'].str.contains(":")
                ]
        
        return_dfs = {}

        if not all_configs: 
            for sweep_name in model_sweep['sweep_name'].unique().tolist():
                subset = model_sweep[model_sweep['sweep_name'] == sweep_name]

                swept_parameters = sweep_name.split(":")[1].split("-")
                model_sweep_parameters = [f"model.{param}" for param in swept_parameters]
                
                avg_df = self.average_across_seed_per_cell_line(
                    data=subset,
                    group_by=model_sweep_parameters
                )

                return_dfs[sweep_name] = avg_df

            return return_dfs
        
        elif all_configs:

            model_sweep_parameters = []

            if not linear: 
                for sweep_name in model_sweep['sweep_name'].unique().tolist():
                    swept_parameters = sweep_name.split(":")[1].split("-")
                    model_sweep_parameters.extend([f"model.{param}" for param in swept_parameters])

            elif not linear: 
                model_sweep_parameters = [
                        'model.l1_ratio', 
                        'model.alpha'
                    ]
                
            subset = model_sweep.sort_values(
                    'holdout_r2_score', 
                    ascending=False
                ).drop_duplicates(
                    subset=model_sweep_parameters + ['dataset.cell_line', 'training.seed'],
                    keep="first"
                )
            logger.warning(f"Dropping model hyperparameter duplicates after all sweeps ran. \nOriginal shape: {model_sweep.shape}, new shape: {subset.shape}")

            avg_df = self.average_across_seed_per_cell_line(
                data=subset,
                group_by=model_sweep_parameters
            )

            self.top_model_configs = avg_df
            return self.top_model_configs
                

                # output_file = "../output/chosen_models_for_outer_loop/top_model_configs_per_cell_line.json"
                # with open(output_file, "w") as f:
                #     json.dump(combined_configs.to_dict(orient="records"), f, indent=4)
                # logger.success(f"Saved top model configurations to {output_file}")
                
                # logger.info(f"Top Model Configurations by Avg. Inner Fold Holdout R2 Score using parameters: {self.performance_chosen_parameters}")
                # self.top_model_configs = combined_configs
                # return self.top_model_configs


    def load_aggreated_data_stats(self):
        
        data_stats_df = pd.read_csv("../output/data_matrix_stats/data_stats.tsv", sep="\t")
        self.matrix_stats_df = data_stats_df

                    
    def plot_r2_distributions(self): 

        for type in self.sweep_results:

            data = self.sweep_results[type]

            plt.figure(figsize=(12, 5), dpi=200)

            if type == 'dataset':
                y_variable = 'val_r2_score'
                y_label = 'Validation R2 Score'
                size=5
                linewidth=1
            elif type == 'model' or type == 'linear':
                y_variable = 'holdout_r2_score'
                y_label = 'Holdout R2 Score'
                size=0.5
                linewidth=0.1

            sns.swarmplot(x='dataset.cell_line', y=y_variable, data=data, palette=['lightblue', 'lightcoral'], linewidth=linewidth, edgecolor='black', size=size)
            sns.boxplot(x='dataset.cell_line', y=y_variable, data=data, showcaps=False, boxprops={'facecolor':'None', 'edgecolor':'black'}, whiskerprops={'color':'black', 'linewidth':2}, medianprops={'color':'black'}, showfliers=False)

            plt.title(f'{type.capitalize()} Sweep: R2 Score Distribution')
            plt.xlabel('Cell Line')
            plt.ylabel(y_label)

            plt.savefig(f"../output/plots/summary/{type}_r2_score_distribution.png", dpi=200, bbox_inches='tight')
            plt.show()
            plt.close()
            

    def plot_dataset_r2_per_covariate(self): 

        if not hasattr(self, 'matrix_stats_df'):
            self.load_aggreated_data_stats()
        
        for covariate in self.dataset_sweep_covariates:
            if covariate in ['dataset.min_read_count', 'dataset.features.binding_matrix.window']:
                fig, axes = plt.subplots(2, 1, figsize=(8, 7), dpi=300, sharex=True, gridspec_kw={'hspace': 0.3})
                ax_top, ax_bottom = axes
            else:
                fig, ax_top = plt.subplots(figsize=(8, 3), dpi=300)
                ax_bottom = None

            sns.swarmplot(x='dataset.cell_line', y='val_r2_score', hue=covariate, data=self.sweep_results['dataset'], dodge=True, palette='Set2', linewidth=1, edgecolor='black', ax=ax_top)
            sns.boxplot(x='dataset.cell_line', y='val_r2_score', hue=covariate, data=self.sweep_results['dataset'], dodge=True, showcaps=False, boxprops={'facecolor':'None', 'edgecolor':'black'}, whiskerprops={'color':'black', 'linewidth':2}, medianprops={'color':'black'}, showfliers=False, ax=ax_top)

            handles, labels = ax_top.get_legend_handles_labels()
            n = len(handles) // 2
            legend_title = self.dataset_sweep_covariates[covariate]
            ax_top.legend(handles[:n], labels[:n], loc='center left', bbox_to_anchor=(1, 0.5), title=legend_title)

            if covariate!='dataset.features.binding_matrix.binding_format':
                ax_top.set_title(f'Validation R2 Score by {legend_title}\nNOTE: includes both binary and expression representation of binding data', y=1)
            else: 
                ax_top.set_title(f'Validation R2 Score by {legend_title}')
                ax_top.set_xlabel('Cell Line')

            ax_top.set_ylabel('Validation R2 Score')            

            if ax_bottom is not None:

                if covariate == 'dataset.min_read_count':
                    y_axis_param = 'validate_count'
                    hue='min_read_count'
                elif covariate == 'dataset.features.binding_matrix.window':
                    y_axis_param = 'avg_binding_per_graph_per_window'
                    hue='window'

                sns.swarmplot(
                    x='cell_line', y=y_axis_param, hue=hue, data=self.matrix_stats_df, palette='Set2', ax=ax_bottom,
                    edgecolor='black', linewidth=1, dodge=True
                )
                sns.boxplot(
                    x='cell_line', y=y_axis_param, hue=hue, data=self.matrix_stats_df, palette='Set2', ax=ax_bottom,
                    showcaps=False, boxprops={'facecolor':'None', 'edgecolor':'black'}, whiskerprops={'color':'black', 'linewidth':2},
                    medianprops={'color':'black'}, showfliers=False, dodge=True
                )
                
                handles, labels = ax_bottom.get_legend_handles_labels()
                n = len(handles) // 2
                ax_bottom.legend(handles[:n], labels[:n], loc='center left', bbox_to_anchor=(1, 0.5), title='Min. Read Count' if hue == 'min_read_count' else 'Window')

                ax_bottom.set_ylabel('# Validation Graphs' if covariate == 'dataset.min_read_count' else 'Avg Binding per Graph \nper Splice Junction')
                ax_bottom.set_xlabel('Cell Line')

                bottom_title_suffix="\nNOTE: includes ONLY binary representation of binding data"
                bottom_title_prefix = "# Validation Graphs vs. Min. Read Count" if covariate == 'dataset.min_read_count' else 'Avg Binding per Graph per Splice Junction vs. Window'
                ax_bottom.set_title(f'{bottom_title_prefix}{bottom_title_suffix}', y=1)

            plt.savefig(f"../output/plots/dataset/dataset_{covariate}_r2_score_distribution.png", dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()


    def plot_1D_range_model_hyperparameter_sweeps(self): 

        model_sweep = self.sweep_results["model"].copy(deep=True)
        model_sweep = model_sweep[model_sweep['sweep_name'].str.startswith("1D_")]

        model_sweep['xgboost_hyperparameter'] = model_sweep['sweep_name'].apply(lambda x: "_".join(x.split("_")[1:]))
        unique_hyperparameters = sorted(model_sweep['xgboost_hyperparameter'].unique())

        for hyperparameter in unique_hyperparameters:
            subset = model_sweep[model_sweep['xgboost_hyperparameter'] == hyperparameter]
            subset = subset.sort_values(by=f"model.{hyperparameter}")

            plt.figure(figsize=(12, 4), dpi=300)
            sns.swarmplot(
                x=f"model.{hyperparameter}", y="holdout_r2_score", hue="dataset.cell_line", 
                data=subset, palette=['red', 'blue'], linewidth=1, edgecolor='black', hue_order=["K562", "HepG2"], dodge=True
            )
            sns.boxplot(
                x=f"model.{hyperparameter}", y="holdout_r2_score", hue="dataset.cell_line", 
                data=subset, showcaps=False, boxprops={'facecolor':'None', 'edgecolor':'black'}, 
                whiskerprops={'color':'black', 'linewidth':2}, medianprops={'color':'black'}, 
                showfliers=False, hue_order=["K562", "HepG2"], dodge=True
            )

            handles, labels = plt.gca().get_legend_handles_labels()
            n = len(handles) // 2
            plt.legend(handles[:n], labels[:n], loc='center left', bbox_to_anchor=(1, 0.5), title='Cell Line')

            plt.title(f'1D Hyperparameter Sweep: "{hyperparameter}"')
            plt.xlabel(f'Value for "{hyperparameter}"')
            plt.ylabel('Holdout R2 Score')

            plt.savefig(f"../output/plots/model/1d_sweeps/model_{hyperparameter}_r2_score_distribution.png", dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

    
    def plot_inner_fold_seed_r2_results(self): 
        for key in ['model', 'linear']:
            if key in self.sweep_results:
                model_sweep = self.sweep_results[key].copy(deep=True)

                model_sweep = model_sweep[model_sweep['training.seed'] != 17]
                logger.warning("REMINDER: Excluding configurations where 'training.seed' is 17")
                
                plt.figure(figsize=(10, 4), dpi=200)

                sns.swarmplot(
                    x='training.seed', y='holdout_r2_score', hue='dataset.cell_line', size=0.8,
                    data=model_sweep, palette=['red', 'blue'], linewidth=0.05, edgecolor='black', hue_order=["K562", "HepG2"], dodge=True
                )
                sns.boxplot(
                    x='training.seed', y='holdout_r2_score', hue='dataset.cell_line', 
                    data=model_sweep, showcaps=False, boxprops={'facecolor':'None', 'edgecolor':'black'}, 
                    whiskerprops={'color':'black', 'linewidth':2}, medianprops={'color':'black'}, 
                    showfliers=False, hue_order=["K562", "HepG2"], dodge=True
                )

                handles, labels = plt.gca().get_legend_handles_labels()
                n = len(handles) // 2
                plt.legend(handles[:n], labels[:n], loc='center left', bbox_to_anchor=(1, 0.5), markerscale=7, title='Cell Line', fontsize=12, title_fontsize=12)

                plt.title(f'Holdout R2 Score by HTD Seed ({key.capitalize()} Sweep)\n(All tuning experiments included)', fontsize=16, y=1.03)
                plt.xlabel('Training Seed', fontsize=14)
                plt.ylabel('Holdout R2 Score', fontsize=14)

                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)

                plt.savefig(f"../output/plots/summary/{key}_inner_fold_seed_r2_results.png", dpi=200, bbox_inches='tight')
                plt.show()
                plt.close()


    def plot_interrelated_model_hyperparameter_results_in_1D(self): 

        model_sweep = self.sweep_results['model'].copy(deep=True)
        model_sweep = model_sweep[model_sweep['sweep_name'].str.contains(":")]

        unique_sweep_names = sorted(model_sweep['sweep_name'].unique())
        
        for sweep_name in unique_sweep_names:
            logger.info(f"Plotting for sweep: {sweep_name}")

            param_names = [f"model.{param}" for param in sweep_name.split(":")[1].split("-")]
            subset = model_sweep[model_sweep['sweep_name'] == sweep_name]

            for param in param_names:
                for hue in ["dataset.cell_line", "training.seed"]: 

                    if hue == "dataset.cell_line":
                        palette = ['red', 'blue']
                        hue_order = ["K562", "HepG2"]
                    elif hue == "training.seed":
                        palette = sns.color_palette("Set2", 9)
                        hue_order = list(range(100, 901, 100))

                    param_name = param.split(".")[-1]
                    plt.figure(figsize=(8,3), dpi=200)
                    
                    sns.swarmplot(
                        x=param, y="holdout_r2_score", hue=hue, size=0.8,
                        data=subset, palette=palette, linewidth=0.05, edgecolor='black', hue_order=hue_order, dodge=True
                    )
                    sns.boxplot(
                        x=param, y="holdout_r2_score", hue=hue, 
                        data=subset, showcaps=False, boxprops={'facecolor':'None', 'edgecolor':'black'}, 
                        whiskerprops={'color':'black', 'linewidth':2}, medianprops={'color':'black'}, 
                        showfliers=False, hue_order=["K562", "HepG2"], dodge=True
                    )

                    handles, labels = plt.gca().get_legend_handles_labels()
                    n = len(subset[hue].unique().tolist())
                    plt.legend(handles[:n], labels[:n], loc='center left', markerscale=4, fontsize=10, ncol=1, bbox_to_anchor=(1, 0.5))


                    plt.title(f"{param_name} in sweep: {', '.join(sweep_name.split(':')[-1].split('-'))}", fontsize=12)
                    plt.xlabel(param_name, fontsize=10)
                    plt.ylabel('Holdout R2 Score', fontsize=10)

                    plt.savefig(
                        f'../output/plots/model/interrelated_hyperparameters/1D_plotting/sweep_{sweep_name.split(":")[0]}-{param_name}-{hue.replace(".", "_")}-r2_score_distribution.png', 
                        dpi=200, 
                        bbox_inches='tight'
                    )
                    plt.show()
                    plt.close()


    def plot_interrelated_model_hyperparameter_results_in_higher_dimensions(self): 

        model_sweep = self.sweep_results['model'].copy(deep=True)
        model_sweep = model_sweep[model_sweep['sweep_name'].str.contains(":")]

        unique_sweep_names = sorted(model_sweep['sweep_name'].unique())
        
        for sweep_name in unique_sweep_names:
            param_names = [f"model.{param}" for param in sweep_name.split(":")[1].split("-")]
            subset = model_sweep[model_sweep['sweep_name'] == sweep_name]

            if "1:" in sweep_name: 
                markerscale = 6
                linewidth=0.1
                size=1.2
            else:
                markerscale = 1
                linewidth=0.5
                size=3
            
            if len(param_names) == 3:
                param_combinations = [(param_names[0], param_names[1]), (param_names[0], param_names[2]), (param_names[1], param_names[2])]
            elif len(param_names) == 2: 
                param_combinations = [(param_names[0], param_names[1])]
            
            for param_x, param_hue in param_combinations:
                fig, axes = plt.subplots(2, 1, figsize=(10, 5), dpi=200, sharex=True, sharey=True)
                cell_lines = subset['dataset.cell_line'].unique()

                for i, cell_line in enumerate(cell_lines):
                    ax = axes[i]
                    cell_line_subset = subset[subset['dataset.cell_line'] == cell_line]

                    hue_order = sorted(cell_line_subset[param_hue].unique())
                    sns.swarmplot(
                        x=param_x, y="holdout_r2_score", hue=param_hue, data=cell_line_subset, size=size, 
                        palette='Set2', linewidth=linewidth, edgecolor='black', ax=ax, dodge=True, hue_order=hue_order
                    )

                    ax.set_title(cell_line, fontsize=16)
                    ax.legend_.remove()  # Remove the legend for each subplot
                    ax.set_xlabel('')
                    ax.set_ylabel('')

                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), markerscale=markerscale, title=param_hue.split(".")[-1], title_fontsize=12)

                plt.suptitle(f'Holdout R2 Score by {param_x.split(".")[-1]} and {param_hue.split(".")[-1]}', y=0.98, fontsize=20)
                plt.tight_layout(rect=[0, 0, 0.98, 1])  # Adjust layout to make space for the legend
                fig.supxlabel(f'{param_x.split(".")[-1]}', fontsize=18, y=-0.06)
                fig.supylabel('Holdout R2 Score', fontsize=18, x=-0.03)
                
                plt.savefig(
                    f"../output/plots/model/interrelated_hyperparameters/higher_dimension_plotting/sweep_{sweep_name.split(':')[0]}-{param_x.split('.')[-1]}-{param_hue.split('.')[-1]}-r2_score_distribution.png",
                    dpi=200,
                    bbox_inches='tight'
                )
                plt.show()
                plt.close()

            if len(param_names) == 3:
                for cell_line in subset['dataset.cell_line'].unique():
                                                    
                    param_x = param_names[0].split(".")[-1]
                    param_y = param_names[1].split(".")[-1]
                    param_z = param_names[2].split(".")[-1]

                    cell_line_subset = subset[(subset['dataset.cell_line'] == cell_line) & (subset['holdout_r2_score'] > 0.2)]

                    fig, axes = plt.subplots(3, 3, figsize=(20, 13), dpi=200, subplot_kw={'projection': '3d'})
                    cmap = plt.cm.rainbow

                    for i, seed in enumerate(sorted(cell_line_subset['training.seed'].unique())):
                        seed_subset = cell_line_subset[cell_line_subset['training.seed'] == seed]

                        ax = axes[i // 3, i % 3]
                        sc = ax.scatter(
                            seed_subset[param_names[0]], seed_subset[param_names[1]], seed_subset[param_names[2]], 
                            c=seed_subset['holdout_r2_score'], cmap=cmap, edgecolor='k', depthshade=False
                        )

                        ax.set_xlabel(param_x)
                        ax.set_ylabel(param_y)
                        ax.set_zlabel(param_z, labelpad=1)
                        ax.set_title(f'Seed: {seed}', pad=10)

                    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.7, pad=0.15, location='right')
                    cbar.ax.set_title('Holdout R2 Score', pad=10)

                    fig.suptitle(f'{cell_line}: Holdout R2 Score by {param_x}, {param_y}, {param_z}\nNOTE: only showing Holdout R2 Scores > 0.2', y=0.98, fontsize=20)
                    
                    plt.savefig(
                        f"../output/plots/model/interrelated_hyperparameters/higher_dimension_plotting/sweep_{sweep_name.split(':')[0]}-{cell_line}-{param_x}-{param_y}-{param_z}-r2_score_distribution.png",
                        dpi=300, 
                        bbox_inches='tight'
                    )
                    plt.show()
                    plt.close()
                    

    def get_run_ids_for_outer_loop_holdout_r2_scores(self): 
        model_sweep = self.sweep_results['model'].copy(deep=True)
        model_sweep = model_sweep[model_sweep['training.seed'] == 100]

        if not hasattr(self, 'top_model_configs'):
            self.show_top_model_configs_after_averaging_by_seed(all_configs=True)

        outer_loop_candidates = pd.concat([
            self.top_model_configs[self.top_model_configs['dataset.cell_line'] == cell_line].head(self.choose_n_configs)
            for cell_line in self.top_model_configs['dataset.cell_line'].unique()
        ], ignore_index=True)

        run_ids = []
        for config in outer_loop_candidates.to_dict(orient="records"):
            subset = model_sweep.copy(deep=True)

            for key, value in config.items():
                if key.startswith("model."): 
                    subset = subset[subset[key] == value]
            
            if subset.shape[0] > 1:
                logger.warning(f"{subset.shape[0]} runs found for config: {config}. \nChoosing the first one after sorting by all columns.")

            subset = subset.sort_values(by=subset.columns.tolist())
            run_ids.append(subset['run_id'].iloc[0])

        run_ids = sorted(run_ids)

        output_file = "../output/chosen_models_for_outer_loop/outer_loop_holdout_run_ids.txt"
        with open(output_file, "w") as f:
            f.write("\n".join(run_ids))

        logger.success(f"Saved run IDs to {output_file}")
        self.run_ids = run_ids
        return self.run_ids


    def assert_outer_loop_holdout_r2_scores(self):
        model_sweep = self.sweep_results['model'].copy(deep=True)

        if not hasattr(self, 'run_ids'):
            self.get_run_ids_for_outer_loop_holdout_r2_scores()

        model_sweep = model_sweep[model_sweep['run_id'].isin(self.run_ids)]

        assert model_sweep['outer_loop_holdout_r2_score'].notna().all(), "Some rows have missing values in the 'outer_loop_holdout_r2_score' column"
        assert model_sweep.shape[0] == len(self.run_ids), "Mismatch between the number of rows in model_sweep and the length of valid_run_ids"
        assert len(model_sweep) == (2 * self.choose_n_configs)
        logger.success("Assertions for outer loop holdout R2 scores passed")

        self.outer_loop_r2_scores_table = model_sweep


    def inner_fold_vs_outer_fold_r2_score(self):

        # Extract all model parameters from sweep_names
        model_sweep = self.sweep_results['model'].copy(deep=True)
        model_sweep = model_sweep[model_sweep['sweep_name'].str.contains(":")]

        model_sweep_parameters = []
        for sweep_name in model_sweep['sweep_name'].unique().tolist():
            swept_parameters = sweep_name.split(":")[1].split("-")
            model_sweep_parameters.extend([f"model.{param}" for param in swept_parameters])
        model_sweep_parameters = list(dict.fromkeys(model_sweep_parameters))  # Remove duplicates while preserving order

        # Merge outer_loop_r2_scores_table with top_model_configs
        merged_table = pd.merge(
            self.outer_loop_r2_scores_table,
            self.top_model_configs,
            on=model_sweep_parameters + ['dataset.cell_line'],
            how='inner',
            suffixes=('_outer', '_inner')
        ).sort_values(
            by=['dataset.cell_line', 'outer_loop_holdout_r2_score', 'avg_holdout_r2_score'],
        )

        assert merged_table.shape[0] == len(self.outer_loop_r2_scores_table), "Mismatch in the number of rows after merging with top_model_configs"    
        self.inner_vs_outer_r2_scores_table = merged_table[model_sweep_parameters + ["run_id", "dataset.cell_line", "avg_holdout_r2_score", "outer_loop_holdout_r2_score"]]
        
        # Create a scatterplot of avg_holdout_r2_score vs. outer_loop_holdout_r2_score
        plt.figure(figsize=(6,3), dpi=200)
        sns.scatterplot(
            x="avg_holdout_r2_score", 
            y="outer_loop_holdout_r2_score", 
            hue="dataset.cell_line", 
            data=self.inner_vs_outer_r2_scores_table, 
            palette="Set2", 
            edgecolor="black", 
            linewidth=1,
            s=40
        )

        # Add the y=x line
        min_val = min(
            self.inner_vs_outer_r2_scores_table["avg_holdout_r2_score"].min(),
            self.inner_vs_outer_r2_scores_table["outer_loop_holdout_r2_score"].min()
        )
        max_val = max(
            self.inner_vs_outer_r2_scores_table["avg_holdout_r2_score"].max(),
            self.inner_vs_outer_r2_scores_table["outer_loop_holdout_r2_score"].max()
        )
        plt.plot([min_val, max_val], [min_val, max_val], color="gray", linestyle="--", linewidth=1, label="y = x")

        plt.title("Inner Fold vs. Outer Loop Holdout R2 Scores", fontsize=14)
        plt.xlabel("Average Holdout R2 Score (Inner Fold)", fontsize=12)
        plt.ylabel("Outer Loop Holdout R2 Score", fontsize=12)
        plt.legend(title="Cell Line", fontsize=9, title_fontsize=10, loc="best")

        # Save the plot
        plt.show()
        plt.close()
        return self.inner_vs_outer_r2_scores_table
    

    def plot_OLS_results(self): 
        linear_sweep = self.sweep_results['linear'].copy(deep=True)
        linear_sweep = linear_sweep[linear_sweep['sweep_name'] == 'OLS']

        # Sort by dataset.cell_line to guarantee x-axis order
        linear_sweep = linear_sweep.sort_values(by='dataset.cell_line')
        unique_cell_lines = linear_sweep['dataset.cell_line'].unique()

        # Extract the exact holdout_r2_score for each cell line
        bar_data = linear_sweep[['dataset.cell_line', 'holdout_r2_score']].drop_duplicates()
        assert bar_data.shape[0] == len(unique_cell_lines), "Mismatch in the number of unique cell lines and exact R2 scores"

        # Create a bar plot for holdout_r2_score by dataset.cell_line
        plt.figure(figsize=(4, 3), dpi=200)
        ax = sns.barplot(
            x='dataset.cell_line', 
            y='holdout_r2_score', 
            data=bar_data, 
            palette='Set2', 
            edgecolor='black', 
            ci=None, 
            width=0.5
        )

        # Extend the y-axis upwards if needed
        max_r2_score = bar_data['holdout_r2_score'].max()
        plt.ylim(0, max_r2_score + 0.05)

        for container in ax.containers: 
            ax.bar_label(container, padding=5)
        # # Add text annotations for each bar
        # for index, row in bar_data.iterrows():
        #     plt.text(
        #         x=index, 
        #         y=row['holdout_r2_score'] + 0.01,  # Position slightly above the bar
        #         s=0.5, 
        #         ha='center', 
        #         va='bottom', 
        #         fontsize=8
        #     )



        plt.title('OLS Holdout R2 Score by Cell Line', fontsize=12)
        plt.xlabel('Cell Line', fontsize=10)
        plt.ylabel('Holdout R2 Score', fontsize=10)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)

        # Save and show the plot
        plt.tight_layout()
        # plt.savefig("../output/plots/linear/ols_r2_score_distribution.png", dpi=200, bbox_inches='tight')
        plt.show()
        plt.close()


    def plot_elasticnet_results(self): 
        linear_sweep = self.sweep_results['linear'].copy(deep=True)
        linear_sweep = linear_sweep[linear_sweep['sweep_name'] == 'ElasticNet']

        unique_cell_lines = linear_sweep['dataset.cell_line'].unique()
        hue_order = sorted(linear_sweep['model.l1_ratio'].unique())

        fig, axes = plt.subplots(1, len(unique_cell_lines), figsize=(20, 6), dpi=200, sharey=True, sharex=True)

        for i, cell_line in enumerate(unique_cell_lines):
            ax = axes[i]
            subset = linear_sweep[linear_sweep['dataset.cell_line'] == cell_line].sort_values(by=['model.alpha', 'model.l1_ratio'])

            sns.swarmplot(
                x='model.alpha', y='holdout_r2_score', hue='model.l1_ratio',
                data=subset, palette='Set2', edgecolor='black', size= 1, linewidth=0.01, ax=ax, hue_order=hue_order, dodge=True
                )

            ax.set_title(f'Cell Line: {cell_line}', fontsize=12)
            ax.legend_.remove()  # Remove legend for individual subplots
            ax.tick_params(axis='x', rotation=90)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))

        # Set figure-level x and y axis labels
        fig.supxlabel('Model Alpha', fontsize=12)
        fig.supylabel('Holdout R2 Score', fontsize=12)

        # Add a single shared legend outside the plot
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), title='L1 Ratio', fontsize=10, title_fontsize=11, markerscale=5)

        plt.tight_layout(rect=[0, 0, 0.99, 1])  # Adjust layout to make space for the legend
        # plt.savefig("../output/plots/linear/elasticnet_r2_score_distribution.png", dpi=200, bbox_inches='tight')
        plt.show()
        plt.close()

        



if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Remove the existing logger
    logger.remove()
    # Reinstantiate the logger to send regular output to stdout and error messages to stderr
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}", filter=lambda record: record["level"].name in ["INFO", "SUCCESS"])
    logger.add(sys.stderr, level="ERROR", format="{time} {level} {message}", filter=lambda record: record["level"].name == "ERROR")

    parser = argparse.ArgumentParser(description="Dataset and Model Parameter Analyzer")
    parser.add_argument('--parallelize', action='store_true', help="Run the analysis in parallel")
    parser.add_argument('--aggregate', action='store_true', help="Aggregate the results")
    parser.add_argument('--cell_line', type=str, help="Specify the cell line for analysis")
    parser.add_argument('--distance', type=int, help="Specify the distance parameter for analysis")
    parser.add_argument('--min_read_count', type=int, help="Specify the minimum read count for analysis")

    args = parser.parse_args()

    if args.parallelize:

        wandb_dataset_sweep = pd.read_csv("../output/wandb_summary_tables/dataset_sweep_summary.tsv", sep="\t")
        unique_combinations = wandb_dataset_sweep[['dataset.cell_line', 'dataset.features.binding_matrix.window', 'dataset.min_read_count']].drop_duplicates()

        cell_lines = sorted(unique_combinations['dataset.cell_line'].unique())
        windows = sorted(unique_combinations['dataset.features.binding_matrix.window'].unique())
        min_read_counts = sorted(unique_combinations['dataset.min_read_count'].unique())

        for cell_line in cell_lines:
            for window in windows:
                for min_read_count in min_read_counts:

                    os.system(
                        f"sbatch --partition=standard --account=platiglab -N1 -n10 --mem=200GB --output=../SLURM_logs/{cell_line}_{window}_{min_read_count}.out --error=../SLURM_logs/{cell_line}_{window}_{min_read_count}.err --wrap='python3.11 dataset_and_model_parameter_chooser.py --cell_line {cell_line} --distance {window} --min_read_count {min_read_count}'"
                    )
        
    elif args.aggregate:

        json_files = glob.glob("../output/data_matrix_stats/*.json")
        data_list = []

        for file in json_files:
            with open(file, 'r') as f:
                data = json.load(f)
                data_list.append(data)

            os.remove(file)

        data_stats_df = pd.DataFrame(data_list)
        data_stats_df.sort_values(by=['cell_line', 'window', 'min_read_count']).to_csv("../output/data_matrix_stats/data_stats.tsv", sep="\t", index=False)
        logger.success("Aggregated data stats and saved to data_stats.tsv")


    else:
        assert args.cell_line is not None, "Cell line must be specified"
        assert args.distance is not None, "Distance (window) must be specified"
        assert args.min_read_count is not None, "Minimum read count must be specified"
        
        analyzer = DatasetAndModelParameterAnalyzer()

        analyzer.calculate_num_examples_and_average_binding(
            args.cell_line, 
            args.distance, 
            args.min_read_count
        )