import json, argparse, sys, gc, copy, pickle, os, gzip
import polars as pl, xgboost as xgb
from loguru import logger
from sklearn.model_selection import train_test_split


_REQUIRED_COLUMNS = ['index', 'ENSEMBL Gene ID', 'Target_PSI']
_DEFAULT_XGBREGRESSOR_KWARGS = {
    'objective': 'reg:logistic',
    'min_child_weight': 1,
    'gamma': 0, 
    'subsample': 1, 
    'colsample_bytree': 1,
    'reg_alpha': 0,
    'reg_lambda': 1, 
    'n_jobs': -1,
    'early_stopping_rounds': 100, 
    'eval_metric': ['rmse', 'logloss'],
}

def run_xgbregressor(data_and_model_args):

    # Load the dataset as a lazyframe
    data = pl.scan_ipc(
        f"../../2_input_binding_exploration/__featherv2-cache__/{data_and_model_args['dataset.cell_line']}_100.feather",
    )
    schema  = data.collect_schema().names()

    # Get the column schema and filter columns
    columns_to_keep = [
            col for col in schema if col.endswith("_binding") 
        ] + _REQUIRED_COLUMNS
    logger.info(f"Columns to keep: {columns_to_keep}")

    # Select only the desired columns
    data = data.select(columns_to_keep).collect()
    # Extract unique ENSEMBL Gene IDs
    unique_genes = data['ENSEMBL Gene ID'].unique().to_list()

    # Split the genes into 90% train and 10% validate
    train_genes, validate_genes = train_test_split(
            unique_genes, 
            train_size=0.9,
            test_size=0.1, 
            random_state=data_and_model_args['random_seed'], 
            shuffle=True,
        )
    assert not set(train_genes).intersection(set(validate_genes)), "Overlap detected between train_genes and validate_genes"
    logger.info(f"Number of train genes: {len(train_genes)}")
    logger.info(f"Number of validate genes: {len(validate_genes)}")

    # Filter the data based on the split
    train_data = data.filter(pl.col('ENSEMBL Gene ID').is_in(train_genes)).to_pandas()
    validate_data = data.filter(pl.col('ENSEMBL Gene ID').is_in(validate_genes)).to_pandas()
    # Assert that there are no overlapping values in the 'index' column
    assert not set(train_data['index']).intersection(set(validate_data['index'])), "Overlap detected between train_data and validate_data in 'index' column"

    # Log the number of rows in train and validate data
    logger.info(f"Number of graphs in train data: {len(train_data)}")
    logger.info(f"Number of graphs in validate data: {len(validate_data)}")

    del data 
    gc.collect()

    train_indices = train_data['index'].to_list()
    validate_indices = validate_data['index'].to_list()
    # Assert that the indices are unique
    assert len(train_indices) == len(set(train_indices)), "Duplicate indices found in train data"
    assert len(validate_indices) == len(set(validate_indices)), "Duplicate indices found in validate data"
    train_indices = set(train_indices)
    validate_indices = set(validate_indices)

    # Separate features and target using pandas
    X_train = train_data.drop(columns=_REQUIRED_COLUMNS)
    y_train = train_data['Target_PSI']
    X_validate = validate_data.drop(columns=_REQUIRED_COLUMNS)
    y_validate = validate_data['Target_PSI']

    # Assert that X_train and X_validate only have column names that end in "_binding"
    assert all(col.endswith("_binding") for col in X_train.columns), "X_train contains columns that do not end with '_binding'"
    assert all(col.endswith("_binding") for col in X_validate.columns), "X_validate contains columns that do not end with '_binding'"
    # Assert that the number of features in X_train and X_validate are the same
    assert X_train.shape[1] == X_validate.shape[1], "X_train and X_validate have different number of features"
    # Assert that the order of columns in X_train and X_validate are the same
    assert list(X_train.columns) == list(X_validate.columns), "Column order in X_train and X_validate do not match"
    
    del train_data, validate_data
    gc.collect()

    # Deep copy the default kwargs for XGBRegressor
    xgb_kwargs = copy.deepcopy(_DEFAULT_XGBREGRESSOR_KWARGS)

    # Update xgb_kwargs with parameters from data_and_model_args
    for key, value in data_and_model_args.items():
        if key.startswith('model.'):
            param_name = key.split('.')[-1]
            xgb_kwargs[param_name] = value
        elif key == 'random_seed':
            xgb_kwargs['random_state'] = value
    
    logger.info(f"XGBRegressor parameters: {xgb_kwargs}")

    # Initialize and train the XGBRegressor model
    model = xgb.XGBRegressor(**xgb_kwargs)
    model.fit(
            X_train, y_train,
            eval_set=[(X_validate, y_validate)],
            verbose=True
        )

    # Save train_indices and validate_indices as attributes to the model
    model.train_indices = train_indices
    model.validate_indices = validate_indices

    # Save the trained model to a gzip-compressed file
    with gzip.open(f"../outputs/pickled_models/{data_and_model_args['unique_id']}.pkl.gz", "wb") as f:
        pickle.dump(model, f)

    logger.success(f"Model saved to ../outputs/pickled_models/{data_and_model_args['unique_id']}.pkl")    


if __name__ == "__main__":

    # Configure loguru to log everything to stdout
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run XGBRegressor model with optional arguments.")
    parser.add_argument('--config_file', type=str, help="Provide JSON config file with data and model arguments.")

    # Parse arguments
    args = parser.parse_args()    
    # Read the JSON config file and load it into a dictionary
    with open(args.config_file, 'r') as f:
        data_and_model_args = json.load(f)

    logger.info(f"Data/model arguments received: {data_and_model_args}")
    run_xgbregressor(data_and_model_args)

    os.remove(args.config_file)
    logger.success(f"Temporary config file {args.config_file} removed.")
    