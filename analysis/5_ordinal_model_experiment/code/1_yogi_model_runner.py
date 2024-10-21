import sys, glob, os, gc, yaml, argparse, wandb
import pandas as pd, polars as pl, numpy as np, matplotlib.pyplot as plt, seaborn as sns

from dataclasses import dataclass, field
from loguru import logger
from tqdm import tqdm
from sklearn.metrics import make_scorer, balanced_accuracy_score, precision_score, mean_absolute_error, median_absolute_error, cohen_kappa_score, matthews_corrcoef
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet
from scipy.sparse import csr_matrix
from sklearn.metrics import get_scorer
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.linear_model import LinearRegression
from collections import OrderedDict
from mpl_toolkits.axes_grid1 import make_axes_locatable

@dataclass
class YogiModelRunner:
    yaml_file_path: str = field(default="")
    wandb_project_name: str = field(default="")
    
    def __post_init__(self):

        WANDB_ENTITY = "platiglab"

        # logger.remove()
        # # TODO: uncomment this line before running on SLURM
        # logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

        with open(self.yaml_file_path, 'r') as file:
            self.kwargs = yaml.safe_load(file)
        
        logger.info(f"Loaded configuration: \n{yaml.dump(self.kwargs, default_flow_style=False, sort_keys=False)}")

        for top_key in self.kwargs.keys():
            for sub_key, value in self.kwargs[top_key].items():
                setattr(self, sub_key, value)

        #TODO remove this line when running on SLURM
        # os.remove(self.yaml_file_path)

        self.wandb_logger = wandb.init(
            entity = WANDB_ENTITY, 
            project=self.wandb_project_name, 
            config = self.kwargs,
        )

        self.load_data()
        self.split_data()
        self.set_scoring_methods()
        self.run_modeling()
        self.plot_predicted_vs_actual()

        self.wandb_logger.finish()

        logger.success(f"COMPLETED SUCCESSFULLY: {self.model} for {self.cell_line} with window size {self.distance}")


    def get_slurm_cpus_per_task(self):
        return int(os.environ.get("SLURM_CPUS_PER_TASK", 1))


    def set_scoring_methods(self): 
        
        logger.info(f"Selecting scoring methods for {self.model}")

        if self.model == "ElasticNetContinuous" or self.model == "OLSRegression":
            self.scoring = {
                'mean_absolute_error': "neg_mean_absolute_error",
                'mean_squared_error': 'neg_mean_squared_error',
                'root_mean_squared_error': 'neg_root_mean_squared_error',
                'median_absolute_error': "neg_median_absolute_error",
                'r2_score': 'r2'
            }

        #TODO add more scores for other models

        # self.scoring = {
        #     'balanced_accuracy': make_scorer(balanced_accuracy_score),
        #     'precision_weighted': make_scorer(precision_score, average='weighted'),
        #     'mean_absolute_error': make_scorer(mean_absolute_error),
        #     'median_absolute_error': make_scorer(median_absolute_error),
        #     'quadratic_cohen_kappa': make_scorer(cohen_kappa_score, weights='quadratic'),
        #     'matthews_corrcoef': make_scorer(matthews_corrcoef)
        # }


    def cast_column_data_types(self, data):

        logger.info(f"Converting columns to appropriate data types for {self.data_flavor} data flavor")

        if self.data_flavor == "binary" or self.data_flavor == "num_peaks":
            for col in self.binding_columns:
                data = data.with_columns(pl.col(col).cast(pl.UInt8))

        elif self.data_flavor == "rbp_exp" or self.data_flavor == "rbp_exp_peak":
            for col in self.binding_columns:
                data = data.with_columns(pl.col(col).cast(pl.Float32))

        return data


    def load_data(self): 

        INPUT_DATA_PATH="../../3_create_RBP_ML_input/4_create_num_peaks_ML_input/final_modeling_input_datasets"
        EXPRESSION_DATA_PATH="../../3_create_RBP_ML_input/2_normalize_raw_counts_matrices/outputs"

        file = glob.glob(f"{INPUT_DATA_PATH}/{self.cell_line}_{self.distance}_*")
        assert len(file)==1, file

        logger.info("Lazy loading dataset for downstream parallel filtering...")

        #TODO remove nrows 
        data = pl.scan_csv(file[0], separator='\t', n_rows=10000)

        if self.read_count_quantile is not None: 

            quantile_threshold = pl.read_csv(
                    file[0], separator='\t', columns=["Total Read Counts"]
                ).select(
                    pl.col("Total Read Counts")
                ).to_series().quantile(self.read_count_quantile)
            
            data = data.filter(pl.col("Total Read Counts") >= quantile_threshold)
            logger.info(f"Only including examples with total read counts > {quantile_threshold} (quantile: {self.read_count_quantile})")

        if self.control_only: 
            logger.info("Only including control samples")
            data = data.filter(pl.col("RBP_KD_Target") == "CTRL")

        if self.psi_subset is not None: 
            logger.info(f"Only including samples with PSI between {self.psi_subset[0]} and {self.psi_subset[1]}")
            data = data.filter(
                (pl.col("Target_PSI") > self.psi_subset[0]) & 
                (pl.col("Target_PSI") < self.psi_subset[1])
            )

        data = data.collect(streaming=True)

        self.binding_columns = [col for col in data.columns if col.endswith("_binding")]
        
        logger.info("Finished filtering steps and gathered dataframe in-memory")
        logger.info(f"Converting data to '''{self.data_flavor}''' data flavor")
        
        if self.data_flavor == "num_peaks" or self.data_flavor =="binary": 

            data = self.cast_column_data_types(data)

            rbp_targets = data.select(pl.col("RBP_KD_Target")).unique().to_series()

            for target in tqdm(rbp_targets, desc="Converting each RBP KD Target's corresponding columns to '0' (e.g. 'RBFOX2 KD' means all 'RBFOX2 feature column' values are set to 0)."):
                target_columns = [col for col in data.columns if col.startswith(f"{target}_")]

                for col in target_columns:
                    data = data.with_columns(
                        pl.when(pl.col("RBP_KD_Target") == target)
                        .then(pl.lit(0))
                        .otherwise(pl.col(col))
                        .alias(col)
                    )
        
        if self.data_flavor =="binary":

            for col in self.binding_columns:
                data = data.with_columns(
                    pl.when(pl.col(col) > 1)
                    .then(pl.lit(1))
                    .otherwise(pl.col(col))
                    .alias(col)
                )
        
        if self.data_flavor =="rbp_exp" or self.data_flavor=="rbp_exp_peak":

            expression_data = glob.glob(f"{EXPRESSION_DATA_PATH}/{self.cell_line}_{self.count_normalization_method}.tsv.gz")
            assert len(expression_data)==1, expression_data
        
            expression_data = pd.read_csv(expression_data[0], sep='\t', index_col=0, compression="gzip")

            if self.log_counts: 
                logger.info("Applying log2 transformation to expression data")
                expression_data = expression_data.map(lambda x: np.log2(x+1))

            expression_data = expression_data.to_dict(orient="dict")

            for col in tqdm(self.binding_columns, desc="Multiplying binding columns by expression data"):
                gene = col.split("_")[0]

                if self.data_flavor=="rbp_exp_peak":
                    data = data.with_columns(
                        pl.when(pl.col(col) > 0)
                        .then(pl.col(col) * pl.col("Sample Name").map_elements(lambda sample: expression_data[sample][gene], return_dtype=pl.Float64))
                        .otherwise(pl.col(col))
                        .alias(col)
                    )
                
                elif self.data_flavor=="rbp_exp":
                    data = data.with_columns(
                        pl.when(pl.col(col) > 0)
                        .then(1 * pl.col("Sample Name").map_elements(lambda sample: expression_data[sample][gene], return_dtype=pl.Float64))
                        .otherwise(pl.col(col))
                        .alias(col)
                    )
            
            data = self.cast_column_data_types(data)
        
        self.full_data = data
        logger.success(f"Data loaded successfully for {self.cell_line} with window size {self.distance} with shape: {self.full_data.shape}")


    def split_data(self):
        logger.info("Splitting data into training testing, and validation sets")

        assert self.data_splitting_method == "gene"

        np.random.seed(self.random_state)
        unique_genes = self.full_data.select("Gene Name").unique().to_series().to_numpy()
        np.random.shuffle(unique_genes)

        train_genes, test_genes, validate_genes = np.split(
            unique_genes, 
            [
                int(len(unique_genes) * self.train_split), 
                int(len(unique_genes) * (self.train_split + self.test_split))
            ]
        )

        train_data = self.full_data.filter(pl.col("Gene Name").is_in(train_genes))
        test_data = self.full_data.filter(pl.col("Gene Name").is_in(test_genes))
        validate_data = self.full_data.filter(pl.col("Gene Name").is_in(validate_genes))

        del self.full_data
        gc.collect()

        datasets = {
            "train": train_data,
            "test": test_data,
            "validate": validate_data
        }

        for split_name, split_data in datasets.items():
            target = self.create_target(split_data)
            input_data = self.create_modeling_input(split_data)

            assert len(target) == input_data.shape[0], f"Length of target ({len(target)}) does not match number of rows in input ({input_data.shape[0]})"

            setattr(self, f"{split_name}_target", target)
            setattr(self, f"{split_name}_input", input_data)

        logger.info(f"Training input shape: {self.train_input.shape}")
        logger.info(f"Testing input shape: {self.test_input.shape}")
        logger.info(f"Validation input shape: {self.validate_input.shape}")

        logger.success("Data successfully split into training, testing, and validation sets")


    def create_target(self, data):

        if self.model == "ElasticNetContinuous" or self.model == "OLSRegression":
            return data["Target_PSI"].to_numpy()
        

    def create_modeling_input(self, data):
        return csr_matrix(data.select(self.binding_columns).to_numpy())


    def run_modeling(self):

        if self.model == "OLSRegression":
            self.run_ols_regression()
        
        elif self.model == "ElasticNetContinuous":
            self.run_elasticnet_linear_regression()


    def run_elasticnet_linear_regression(self):

        logger.info(f"Running ElasticNet linear regression for {self.cell_line} with window size {self.distance}")


        def evaluate_model(alpha, l1_ratio):
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=self.random_state)
            model.fit(self.train_input, self.train_target)

            train_r2 = model.score(self.train_input, self.train_target)
            test_r2 = model.score(self.test_input, self.test_target)
            validate_r2 = model.score(self.validate_input, self.validate_target)

            avg_r2 = (test_r2 + validate_r2) / 2

            logger.info(f"Alpha: {alpha}, L1 Ratio: {l1_ratio}, Train R2: {train_r2}, Test R2: {test_r2}, Validate R2: {validate_r2}, Avg R2: {avg_r2}")

            return avg_r2, alpha, l1_ratio, train_r2, test_r2, validate_r2

        results = []
        with ThreadPoolExecutor(max_workers=self.get_slurm_cpus_per_task()) as executor:
            futures = {executor.submit(evaluate_model, alpha, l1_ratio): (alpha, l1_ratio) for alpha in self.alpha for l1_ratio in self.l1_ratio}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating models", unit="model"):
                results.append(future.result())

        best_score, best_alpha, best_l1_ratio, best_train_r2, best_test_r2, best_validate_r2 = max(results, key=lambda x: x[0])

        logger.success(
            f"Results from best run ---- Alpha: {best_alpha}, L1 Ratio: {best_l1_ratio}, Avg R2: {best_score}, Train R2: {best_train_r2}, Test R2: {best_test_r2}, Validate R2: {best_validate_r2}"
        )

        # Run the ElasticNet model again with the best parameters
        self.final_model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, random_state=self.random_state)
        self.final_model.fit(self.train_input, self.train_target)

        self.log_scoring()

        logger.success("Final model evaluation completed with the best parameters")


    def run_ols_regression(self):
        
        logger.info(f"Running OLS regression for {self.cell_line} with window size {self.distance}")

        self.final_model = LinearRegression(
            n_jobs=self.get_slurm_cpus_per_task(), 
            copy_X=False
        )

        self.final_model.fit(self.train_input, self.train_target)

        self.log_scoring()

        logger.success("Model evaluation completed")


    def log_scoring(self):

        scoring_dict = OrderedDict()

        for score_name, score_func_name in self.scoring.items():
            score_func = get_scorer(score_func_name)

            train_score = score_func(self.final_model, self.train_input, self.train_target)
            test_score = score_func(self.final_model, self.test_input, self.test_target)
            validate_score = score_func(self.final_model, self.validate_input, self.validate_target)

            logger.info(f"Model scoring: {score_name} - Train: {train_score}, Test: {test_score}, Validate: {validate_score}")

            scoring_dict[f"train_{score_name}"] = train_score
            scoring_dict[f"test_{score_name}"] = test_score
            scoring_dict[f"validate_{score_name}"] = validate_score

        self.wandb_logger.log(scoring_dict)


    def plot_predicted_vs_actual(self):
            
        logger.info("Plotting predicted vs actual values")

        train_predictions = self.final_model.predict(self.train_input)
        test_predictions = self.final_model.predict(self.test_input)
        validate_predictions = self.final_model.predict(self.validate_input)

        if self.model == "ElasticNetContinuous" or self.model == "OLSRegression":

            ##############################################
            # Plotting Actual vs Predicted 2D Histograms #
            ##############################################

            fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=100, sharex=True, sharey=True)

            for ax, (actual, predicted, dataset_name) in zip(axs, [
                (self.train_target, train_predictions, f"Train ({self.train_split * 100}%)"),
                (self.test_target, test_predictions, f"Test ({self.test_split * 100}%)"),
                (self.validate_target, validate_predictions, f"Validate ({self.validate_split * 100}%)"),
            ]):
                
                assert len(actual) == len(predicted), f"Length of actual ({len(actual)}) does not match length of predicted ({len(predicted)})"

                hist = sns.histplot(x=actual, y=predicted, bins=100, cmap="YlOrBr", ax=ax, stat="percent")
                ax.set_title(f'{dataset_name}', fontsize=26, pad=20)

            # Create a colorbar legend
            cbar = fig.colorbar(hist.collections[0], ax=axs, cax=fig.add_axes([1.05, 0.1, 0.02, 0.8]))
            cbar.set_label('Percentage', fontsize=16)

            fig.suptitle("2D Accuracy Histograms", fontsize = 36, y=1.02)
            fig.supxlabel('Actual', fontsize=28)
            fig.supylabel('Predicted', fontsize=28, x=-0.02)

            self.wandb_logger.log({"actual_vs_predicted_2d_hist": wandb.Image(fig)})

            #########################################################
            # Plot Actual and Predicted Distributions Independently #
            #########################################################

            fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=100, sharex=True, sharey=True)

            # Determine the range for the bins
            min_value = min(
                min(self.train_target.min(), train_predictions.min()),
                min(self.test_target.min(), test_predictions.min()),
                min(self.validate_target.min(), validate_predictions.min())
            )
            max_value = max(
                max(self.train_target.max(), train_predictions.max()),
                max(self.test_target.max(), test_predictions.max()),
                max(self.validate_target.max(), validate_predictions.max())
            )

            # Create 100 evenly spaced bins
            bins = np.linspace(min_value, max_value, 100)

            for ax, (actual, predicted, dataset_name) in zip(axs, [
                (self.train_target, train_predictions, f"Train ({self.train_split * 100}%)"),
                (self.test_target, test_predictions, f"Test ({self.test_split * 100}%)"),
                (self.validate_target, validate_predictions, f"Validate ({self.validate_split * 100}%)"),
            ]):
                
                assert len(actual) == len(predicted), f"Length of actual ({len(actual)}) does not match length of predicted ({len(predicted)})"

                sns.histplot(actual, bins=bins, color="blue", kde=False, stat="percent", ax=ax, label="Actual", alpha=0.5, edgecolor="black", linewidth=0.8)
                sns.histplot(predicted, bins=bins, color="orange", kde=False, stat="percent", ax=ax, label="Predicted", alpha=0.5, edgecolor="black", linewidth=0.8)

                ax.set_title(f'{dataset_name}', fontsize=22, pad=20)

            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)

            fig.suptitle("PSI Distributions for Actual and Predicted", fontsize = 28, y=1.02)
            fig.supxlabel('PSI Values', fontsize=20)
            fig.supylabel('Percentage', fontsize=20, x=-0.02)

            self.wandb_logger.log({"distribution_comparison": wandb.Image(fig)})

            #########################################################################
            # Plotting Distribution of Actual Values Subtracted by Predicted Values #
            #########################################################################

            fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=100, sharey=True, sharex=True)

            # Calculate the min and max values of the differences across all sets
            all_differences = np.concatenate([
                self.train_target - train_predictions,
                self.test_target - test_predictions,
                self.validate_target - validate_predictions
            ])
            min_diff = all_differences.min()
            max_diff = all_differences.max()

            # Create 100 custom bins spanning the min and max value
            bins = np.linspace(min_diff, max_diff, 100)

            for ax, (actual, predicted, dataset_name) in zip(axs, [
                (self.train_target, train_predictions, f"Train ({self.train_split * 100}%)"),
                (self.test_target, test_predictions, f"Test ({self.test_split * 100}%)"),
                (self.validate_target, validate_predictions, f"Validate ({self.validate_split * 100}%)"),
            ]):
                
                assert len(actual) == len(predicted), f"Length of actual ({len(actual)}) does not match length of predicted ({len(predicted)})"

                differences = actual - predicted
                ax.hist(differences, bins=bins, density=True, color='lime', edgecolor='black', linewidth=1)

                ax.set_title(f'{dataset_name}', fontsize=20, pad=20)

            fig.suptitle('Distribution of (Actual-Predicted)', fontsize=24, y=1.02)
            fig.supxlabel('Difference (Actual - Predicted)', fontsize=18)
            fig.supylabel('Percent', fontsize=18, x=-0.02)

            self.wandb_logger.log({"actual_minus_predicted_hist": wandb.Image(fig)})

        logger.success("Plotted predicted vs actual values")

#TODO get ordinal modeling running later 
#     def run_ordinal_modeling(self):

#         logger.info(f"Running cross-validation for model type: {self.model}")

#         skf = StratifiedKFold(n_splits=self.splits, shuffle=self.training_shuffle, random_state=self.random_state)

#         binding_input = csr_matrix(self.full_data.select(self.binding_columns).to_numpy())
#         prediction_target= self.full_data["Ordinal Target"].to_numpy()

#         logger.info(f"Length of prediction_target: {len(prediction_target)}")

#         del self.full_data
#         gc.collect()

#         # Use the wrapper class as the estimator
#         scores = cross_validate(
#             estimator=ElasticNetCVWrapper(),
#             X=binding_input,
#             y=prediction_target,
#             cv=skf,
#             scoring="neg_root_mean_squared_error",
#             verbose=6,
#             n_jobs=self.get_slurm_cpus_per_task(),
#         )

#         # # Use the wrapper class as the estimator
#         # scores = cross_validate(
#         #     estimator=OrderedModelWrapper(method=self.ordinal_regression_method, optimizer=self.optimizer),
#         #     X=binding_input,
#         #     y=prediction_target,
#         #     cv=skf,
#         #     scoring=self.scoring,
#         #     verbose=6,
#         #     n_jobs=self.get_slurm_cpus_per_task(),
#         # )

#         self.model_output = scores
#         logger.success(f"COMPLETED: Cross-validation completed for {self.model}")
        

# class OrderedModelWrapper(BaseEstimator, RegressorMixin):

#     def __init__(self, method=None, optimizer=None):
#         self.method = method
#         self.optimizer = optimizer
#         self.model_ = None

#     def fit(self, X, y):
#         self.model_ = OrderedModel(y, X, distr=self.method)
#         self.result_ = self.model_.fit(method=self.optimizer, disp=False)
#         return self

#     def predict(self, X):
#         return self.result_.predict(X).argmax(axis=1)

#     # def score(self, X, y):
#     #     predictions = self.predict(X)
#     #     return np.mean(predictions == y)

# class ElasticNetCVWrapper(BaseEstimator, RegressorMixin):

#     def __init__(self, **kwargs):
#         self.model = ElasticNetCV(**kwargs)

#     def fit(self, X, y):
#         self.unique_y = np.unique(y)
#         self.model.fit(X, y)
#         return self

#     def predict(self, X):
#         predictions = self.model.predict(X)
#         rounded_predictions = np.round(predictions)
#         clipped_predictions = np.clip(rounded_predictions, np.min(self.unique_y.min()), np.max(self.unique_y.max()))
#         return clipped_predictions
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YogiModelRunner with a specified YAML configuration file and Weights & Biases Project name.")

    parser.add_argument('--yaml-file-path', type=str, required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--wandb-project-name', type=str, required=True, help='Name of the Weights & Biases project.')

    args = parser.parse_args()

    runner = YogiModelRunner(
        yaml_file_path=args.yaml_file_path, 
        wandb_project_name=args.wandb_project_name
    )
