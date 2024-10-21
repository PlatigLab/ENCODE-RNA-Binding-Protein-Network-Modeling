import sys, pandas as pd, polars as pl, glob, numpy as np, os, json, time, gc

from dataclasses import dataclass, field
from loguru import logger
from tqdm import tqdm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, balanced_accuracy_score, precision_score, mean_absolute_error, median_absolute_error, cohen_kappa_score, matthews_corrcoef
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNetCV
from scipy.sparse import csr_matrix


@dataclass
class YogiModelRunner:
    yaml_file_path: str = field(default="")
    
    def __post_init__(self):

        logger.remove()
        # TODO: uncomment this line before running on SLURM
        logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

        with open(self.yaml_file_path, 'r') as file:
            self.kwargs = yaml.safe_load(file)
        
        logger.info(f"Loaded configuration: \n{yaml.dump(self.kwargs, default_flow_style=False, sort_keys=False)}")

        for top_key in self.kwargs.keys():
            for sub_key, value in self.kwargs[top_key].items():
                setattr(self, sub_key, value)

        #TODO remove this line when running on SLURM
        # os.remove(self.yaml_file_path)

        self.load_data()
        self.split_data()
        self.set_scoring_methods()
        self.run_modeling()

        logger.success(f"COMPLETED SUCCESSFULLY: {self.model} for {self.cell_line} with window size {self.distance}")


    def get_slurm_cpus_per_task(self):
        return int(os.environ.get("SLURM_CPUS_PER_TASK", 1))


    def set_scoring_methods(self): 
        
        logger.info(f"Selecting scoring methods for {self.model}")

        if self.model == "ElasticNetContinuous":
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
        EXPRESSION_DATA_PATH="../../3_create_RBP_ML_input/2_normalize_raw_counts_matrices/outputs/"

        file = glob.glob(f"{INPUT_DATA_PATH}/{self.cell_line}_{self.distance}_*")
        assert len(file)==1, file

        logger.info("Lazy loading dataset for downstream parallel filtering...")

        #TODO remove nrows 
        data = pl.scan_csv(file[0], separator='\t', )

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
        self.metadata_columns = [col for col in data.columns if col not in self.binding_columns]

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

        if self.model == "ElasticNetContinuous":
            return data["Target_PSI"].to_numpy()
        

    def create_modeling_input(self, data):
        return csr_matrix(data.select(self.binding_columns).to_numpy())


    def run_modeling(self):

        if self.model == "ElasticNetContinuous":
            self.run_elasticnet_linear_regression()



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

    runner = YogiModelRunner(yaml_file_path=args.yaml_file_path)
