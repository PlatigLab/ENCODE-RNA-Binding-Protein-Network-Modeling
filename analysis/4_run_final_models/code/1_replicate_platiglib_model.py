import json, argparse, sys, gc, copy, pickle, os, gzip, glob

import polars as pl

from loguru import logger
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score

CONFIGS_DIR = "../../3_choose_dataset_and_model_parameters/output/model_reproduction"
MODEL_DIR = "../outputs/pickled_models"
PREDICTIONS_DIR = "../outputs/predictions"

@dataclass
class YogiPlatigLibModelReplicator:
    config_file: str

    DATA_DIR = "../../2_input_binding_exploration/__featherv2-cache__/"
    _REQUIRED_COLUMNS = ['index', 'Target_PSI']


    def __post_init__(self):
        self.initialize_config()
    

    def initialize_config(self):
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)
        
        logger.info(f"Loaded config file: {self.config_file}")
        logger.info(f"Config: {self.config}")

        self.cell_line = self.config.pop('cell_line')
        self.model_type = self.config.pop('name')
        self.config['random_state'] = self.config.pop('seed')

        with gzip.open(f"{CONFIGS_DIR}/rows_to_reproduce/{self.cell_line}.pkl.gz", 'rb') as f:
            self.indices_dict = pickle.load(f)


    def replicate_model(self): 
        self.get_unique_ids_for_training_and_evaluation() 
        self.get_training_validation_test_data()
        self.run_model()   
        self.get_performance()
        self.get_all_predictions()
        self.save_model_and_predictions()
    

    def get_unique_ids_for_training_and_evaluation(self): 

        unique_ids_dict = {}
        for key, indices in self.indices_dict.items():

            if key.endswith("_ind"):
                metadata_df = self.indices_dict["metadata_df"]
                subset_df = metadata_df[metadata_df['index'].isin(indices)]

                assert len(subset_df) == len(indices), f"Length mismatch for {key}: {len(subset_df)} vs {len(indices)}"
                unique_ids_dict[key] = set(subset_df["unique_id"])
        
        self.unique_ids_dict = unique_ids_dict
        del self.indices_dict
        gc.collect()

    
    def get_training_validation_test_data(self):

        logger.info(f"Getting data splits with unique IDs for {self.unique_ids_dict.keys()}")

        for key, unique_ids in self.unique_ids_dict.items():
            df = pl.scan_ipc(
                    f"{self.DATA_DIR}/{self.cell_line}_100.feather", 
                )
            
            # Get the column schema of the lazyframe
            column_schema = df.collect_schema()
            
            # Select columns that end in "_binding" and the required columns
            selected_columns = [
                col for col in column_schema.keys() 
                if col.endswith("_binding") or col in self._REQUIRED_COLUMNS
            ]
            
            # Apply the selection, filter the dataframe, and sort by the index column
            df = df.select(selected_columns).filter(
                    pl.col("index").is_in(unique_ids)
                ).sort("index").collect()
            
            assert len(df) == len(unique_ids), f"Length mismatch for {key}: {len(df)} vs {len(unique_ids)}"

            # Separate input data and evaluation column
            evaluation_data = df.select(["Target_PSI"]).to_pandas()
            df = df.select([col for col in df.columns if col.endswith("_binding")]).to_pandas()

            # Store the splits in attributes for later use
            setattr(self, f'{key.split("_")[0]}_input', df)
            setattr(self, f'{key.split("_")[0]}_target', evaluation_data)

            logger.info(f"Data for {key} has {len(df)} rows and {len(df.columns)} columns.")

        logger.success(f"Data splits for {self.unique_ids_dict.keys()} are ready.")


    def run_model(self):

        # Dynamically get the model class from the string name
        model_class = globals().get(self.model_type)
        assert model_class is not None, f"Model class {self.model_type} not found."

        # Initialize the model with the parameters from the config
        model = model_class(**self.config)
        logger.info(f"{model_class} instantiated with parameters: {model.get_params()}")

        # Assert that all columns in train_input end with "_binding"
        assert all(col.endswith("_binding") for col in self.train_input.columns), "Not all columns in train_input end with '_binding'"
        
        if hasattr(self, "validate_input"):
            assert self.model_type == "XGBRegressor", f"Model type must be XGBRegressor when validate_data is present, but got {self.model_type}"
            # Assert that train_input and validate_input have the same column order
            assert list(self.train_input.columns) == list(self.validate_input.columns), "train_input and validate_input must have the same column order"

            logger.info(f"Training shape {self.train_input.shape}, Validation shape {self.validate_input.shape}")

            # Get the training and evaluation data
            model = model.fit(
                self.train_input, 
                self.train_target, 
                eval_set=[(self.validate_input, self.validate_target)],
            ) 
        else: 
            logger.info(f"Training shape {self.train_input.shape}")
            model = model.fit(
                self.train_input, 
                self.train_target
            )

        model.column_order_when_fitting = self.train_input.columns
        self.model = model

        logger.success(f"{model_class} model fitted.")

    
    def get_performance(self):

        # Use the model to predict on the test data
        test_predictions = self.model.predict(self.test_input)
        # Calculate the R2 score
        r2 = r2_score(self.test_target, test_predictions)
        # Log the performance
        logger.info(f"Model R2 score on test data: {r2}")

    
    def get_all_predictions(self): 

        # Delete all attributes of self that end in _input or _target
        attributes_to_delete = [
            attr for attr in dir(self) if attr.endswith("_input") or attr.endswith("_target")
        ]
    
        for attr in attributes_to_delete:
            delattr(self, attr)
        gc.collect()

        # Read the feather file again to get all data
        df = pl.scan_ipc(f"{self.DATA_DIR}/{self.cell_line}_100.feather").sort("index").collect()4
        original_df_size = len(df)

        all_target_data = df.select(["Target_PSI"]).to_pandas()
        all_input_data = df.select(
            [col for col in df.columns if col.endswith("_binding")]
        ).to_pandas()

        assert len(all_input_data) == len(all_target_data), f"Length mismatch: {len(all_input_data)} vs {len(all_target_data)}"
        # Ensure the column order matches the order used during model fitting
        assert list(all_input_data.columns) == list(self.model.column_order_when_fitting), \
            f"Column order mismatch: {list(all_input_data.columns)} vs {list(self.model.column_order_when_fitting)}"

        # Use the model to predict on the entire dataset
        predictions = self.model.predict(all_input_data)
        # Add predictions to the original dataframe
        df = df.with_columns(pl.Series("Predictions", predictions))

        logger.info(f"R2 score for all data predictions: {r2_score(all_target_data, predictions)}")

        # Add a new column "Partition" to specify Train, Test, or Validate
        partition_column = []
        for idx in df["index"]:
            partition = None
            for key, unique_ids in self.unique_ids_dict.items():
                if idx in unique_ids:
                    partition = key.split("_")[0].capitalize()
                    break

            assert partition is not None, f"Index {idx} not found in any partition."
            partition_column.append(partition)

        df = df.with_columns(pl.Series("Partition", partition_column))
        # Assert that there are no missing values in the Partition column
        assert df["Partition"].is_not_null().all(), "Partition column contains missing values."

        assert len(df) == original_df_size, f"Length mismatch after adding predictions: {len(df)} vs {original_df_size}"
        self.output_df = df
        logger.success(f"All predictions made.")


    def save_model_and_predictions(self):
        
        for dir in [MODEL_DIR, PREDICTIONS_DIR]:
            os.makedirs(f"{dir}/{self.model_type}/", exist_ok=True)

        # Save the model
        model_filename = f"{MODEL_DIR}/{self.model_type}/{self.cell_line}.pkl.gz"
        with gzip.open(model_filename, 'wb') as f:
            pickle.dump(self.model, f)

        predictions_filename = f"{PREDICTIONS_DIR}/{self.model_type}/{self.cell_line}_predictions.feather"   
        self.output_df.write_ipc(predictions_filename, compression="lz4")

        logger.success(f"Model and predictions saved to {MODEL_DIR}/{self.model_type}/ and {PREDICTIONS_DIR}/{self.model_type}/ respectively.")


if __name__ == "__main__":

    # Configure loguru to log everything to stdout
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run XGBRegressor model with optional arguments.")
    parser.add_argument('--config_file', type=str, help="Provide JSON config file with data and model arguments.")
    parser.add_argument('--run_all', action='store_true', help="Run all JSON config files (1 job for each config).")

    # Parse arguments
    args = parser.parse_args() 

    if args.run_all:

        CPUS = 32
        MEM= 256
        PARTITION="standard"
        ACCOUNT="platiglab"
        SLURM_DIR="../outputs/SLURM_logs/"

        # Get all JSON config files in the directory
        json_files = sorted(glob.glob(f"{CONFIGS_DIR}/model_parameters/*.json"))

        # Iterate over each JSON file and run the model
        for json_file in json_files:
            
            job_prefix = json_file.split("/")[-1].split(".")[0]

            os.system(
                f"sbatch --job-name={job_prefix} -n{CPUS} --mem={MEM}GB --partition={PARTITION} --account={ACCOUNT} --output={SLURM_DIR}/{job_prefix}.out --error={SLURM_DIR}/{job_prefix}.err --wrap='/bin/python3.11 {__file__} --config_file {json_file}'"
            )

            sys.exit(0)

    elif args.config_file:
        YogiPlatigLibModelReplicator(args.config_file).replicate_model()
