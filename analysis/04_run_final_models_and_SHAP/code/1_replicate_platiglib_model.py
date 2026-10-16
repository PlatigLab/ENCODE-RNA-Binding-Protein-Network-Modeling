import json, argparse, sys, gc, pickle, os, gzip, glob

import polars as pl

from loguru import logger
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score

CONFIGS_DIR = "../../03_choose_dataset_and_model_parameters/output/model_reproduction"
DATA_SPLITS_DIR = "../outputs/data_splits"
MODEL_DIR = "../outputs/pickled_models"
PREDICTIONS_DIR = "../outputs/predictions"

@dataclass
class YogiPlatigLibModelReplicator:
    config_file: str

    DATA_DIR = "../../02_input_binding_exploration/__featherv2-cache__/"
    _REQUIRED_COLUMNS = ['index', 'Target_PSI']


    def __post_init__(self):

        if self.config_file == "Split Data":
            self.split_data()
        elif self.config_file.endswith(".json"):
            self.initialize_config()
        else: 
            raise ValueError("Invalid config_file argument. Must be 'Split Data' or a path to a JSON config file.")
    

    def split_data(self):
        
        for cell_line in ["HepG2", "K562"]: 
            OUTPUT_FILE = f"{DATA_SPLITS_DIR}/{cell_line}_splits.json.gz"

            cell_line_lf = pl.scan_ipc(
                    f"{self.DATA_DIR}/{cell_line}_100.feather", 
                )
            # Get unique ENSEMBL Gene IDs
            gene_ids = sorted(cell_line_lf.select("ENSEMBL Gene ID").unique().collect()["ENSEMBL Gene ID"].to_list())

            # Split genes into train (80%) and test (20%)
            train_genes, test_genes = train_test_split(
                gene_ids, 
                test_size=0.2, 
                random_state=17
            )

            # Split train genes into train (90%) and validate (10%)
            train_genes, validate_genes = train_test_split(
                train_genes, 
                test_size=0.1, 
                random_state=17
            )

            # Get index values for each set
            train_indices = cell_line_lf.filter(
                pl.col("ENSEMBL Gene ID").is_in(train_genes)
            ).select("index").collect()["index"].sort().to_list()

            test_indices = cell_line_lf.filter(
                pl.col("ENSEMBL Gene ID").is_in(test_genes)
            ).select("index").collect()["index"].sort().to_list()

            validate_indices = cell_line_lf.filter(
                pl.col("ENSEMBL Gene ID").is_in(validate_genes)
            ).select("index").collect()["index"].sort().to_list()

            # Assert no overlap between sets
            assert len(set(train_indices) & set(test_indices)) == 0, "Overlap between train and test"
            assert len(set(train_indices) & set(validate_indices)) == 0, "Overlap between train and validate"
            assert len(set(test_indices) & set(validate_indices)) == 0, "Overlap between test and validate"
            assert len(train_indices) + len(test_indices) + len(validate_indices) == cell_line_lf.select(pl.col('index').count()).collect().item(), "Total indices do not match original dataframe length"

            # Create and save dictionary
            splits_dict = {
                "train_ind": sorted(list(train_indices)),
                "test_ind": sorted(list(test_indices)),
                "validate_ind": sorted(list(validate_indices)),
            }

            # if there's a previous splits file for this cell line, assert that the splits are the same
            if os.path.exists(OUTPUT_FILE):
                with gzip.open(OUTPUT_FILE, 'rt') as f:
                    previous_splits_dict = json.load(f)

                for key in splits_dict.keys():
                    assert set(splits_dict[key]) == set(previous_splits_dict[key]), f"Splits for {cell_line} have changed since the last time they were generated. Please investigate before proceeding. Key with mismatch: {key}"

            with gzip.open(OUTPUT_FILE, 'wt') as f:
                json.dump(splits_dict, f, indent=2)

            logger.info(f"Saved splits for {cell_line}: train={len(train_indices)}, test={len(test_indices)}, validate={len(validate_indices)}")
    
        logger.success("Data splitting completed for all cell lines.")


    def initialize_config(self):
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)

        self.unique_hash = self.config_file.split("/")[-1].split(".")[0]
        
        logger.info(f"Loaded config file: {self.config_file}")
        logger.info(f"Config: {self.config}")

        self.cell_line = self.config.pop('cell_line')
        self.model_type = self.config.pop('name')
        self.config['random_state'] = self.config.pop('seed')


    def replicate_model(self): 
        self.get_unique_ids_for_training_and_evaluation() 
        self.get_training_validation_test_data()
        self.run_model()   
        self.get_performance()
        self.get_all_predictions()
        self.save_model_and_predictions()

        logger.success(f"MODEL REPLICATION COMPLETED\nCell line: {self.cell_line}\nModel type: {self.model_type}\nConfig: {self.config}")
    

    def get_unique_ids_for_training_and_evaluation(self): 

        with gzip.open(f"{DATA_SPLITS_DIR}/{self.cell_line}_splits.json.gz", 'rt') as f:
            indices_dict = json.load(f)
        
        if self.model_type == "XGBRegressor":
            self.unique_ids_dict = {
                "train_ind": indices_dict["train_ind"],
                "test_ind": indices_dict["test_ind"],
                "validate_ind": indices_dict["validate_ind"],
            }
        
        elif self.model_type == "ElasticNet":
            self.unique_ids_dict = {
                "train_ind": indices_dict["train_ind"] + indices_dict["validate_ind"],
                "test_ind": indices_dict["test_ind"],
            }

    
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
        # Assert that train_input and test_input have the same column order
        assert list(self.train_input.columns) == list(self.test_input.columns), "train_input and test_input must have the same column order"
        
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
        self.test_r2_score = r2

    
    def get_all_predictions(self): 

        # Delete all attributes of self that end in _input or _target
        attributes_to_delete = [
            attr for attr in dir(self) if attr.endswith("_input") or attr.endswith("_target")
        ]
        for attr in attributes_to_delete:
            delattr(self, attr)

        gc.collect()

        # Read the feather file again to get all data
        df = pl.scan_ipc(f"{self.DATA_DIR}/{self.cell_line}_100.feather").sort("index").collect()
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
        # Assert that there are no null values in the Predictions column
        assert df["Predictions"].is_not_null().all(), "Predictions column contains null values."

        logger.info(f"R2 score for all data predictions: {r2_score(all_target_data, predictions)}")

        # Create a mapping of index to partition
        partition_data = []
        for key, unique_ids in self.unique_ids_dict.items():
            partition_name = key.split("_")[0].capitalize()
            for idx in unique_ids:
                partition_data.append({"index": idx, "Partition": partition_name})
        
        partition_df = pl.DataFrame(partition_data)
        
        df = df.join(partition_df, on="index", how="left")

        assert df["Partition"].is_in({"Train", "Test", "Validate"}).all(ignore_nulls=False), "Partition column contains unexpected values."
        assert len(df) == original_df_size, f"Length mismatch after adding predictions: {len(df)} vs {original_df_size}"
        
        self.output_df = df
        logger.success(f"All predictions made.")


    def save_model_and_predictions(self):
        
        for dir in [MODEL_DIR, PREDICTIONS_DIR]:
            os.makedirs(f"{dir}/{self.model_type}/", exist_ok=True)

        # Save the model
        model_filename = f"{MODEL_DIR}/{self.model_type}/{self.unique_hash}.pkl.gz"
        with gzip.open(model_filename, 'wb') as f:
            pickle.dump(self.model, f)

        predictions_filename = f"{PREDICTIONS_DIR}/{self.model_type}/{self.unique_hash}.feather"   
        self.output_df.write_ipc(predictions_filename, compression="lz4")

        with open(f"{PREDICTIONS_DIR}/{self.model_type}/{self.unique_hash}_test_r2_score.txt", 'w') as f:
            f.write(f"{self.test_r2_score}\n")

        logger.success(f"Model and predictions saved to {MODEL_DIR}/{self.model_type}/ and {PREDICTIONS_DIR}/{self.model_type}/ respectively.")


if __name__ == "__main__":

    # Configure loguru to log everything to stdout
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run XGBRegressor model with optional arguments.")
    parser.add_argument('--config_file', type=str, help="Provide JSON config file with data and model arguments.")
    parser.add_argument('--run_all', action='store_true', help="Run all JSON config files (1 job for each config).")
    parser.add_argument('--split_data', action='store_true', help="Create the train/test/validate splits. This NEEDS TO BE RUN FIRST before running the model replication with --config_file or --run_all, but only needs to be run once.")

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

            if len(glob.glob(f"{PREDICTIONS_DIR}/**/{job_prefix}*.feather", recursive=True)) == 0: 
                os.system(
                    f"sbatch --job-name={job_prefix} -n{CPUS} --mem={MEM}GB --partition={PARTITION} --account={ACCOUNT} --output={SLURM_DIR}/{job_prefix}.out --error={SLURM_DIR}/{job_prefix}.err --wrap='/bin/python3.11 {__file__} --config_file {json_file}'"
                )
            else: 
                print(f"Skipping hash {job_prefix} as it has already been run.")

    elif args.split_data:
        YogiPlatigLibModelReplicator("Split Data")   

    elif args.config_file:
        YogiPlatigLibModelReplicator(args.config_file).replicate_model()
