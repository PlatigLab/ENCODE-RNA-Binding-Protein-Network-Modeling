import sys, os, argparse, glob, gzip, pickle, shap
import polars as pl, pandas as pd
from loguru import logger

MODEL_DIR = "../outputs/pickled_models/XGBRegressor/"
PREDICTIONS_DIR = "../outputs/predictions/XGBRegressor/"
SHAP_DIR = "../outputs/SHAP/"
SLURM_DIR="../outputs/SLURM_logs/"

CPUS = 32
MEM= 256
PARTITION="standard"
ACCOUNT="platiglab"


def main(hash, normal_or_interaction):
    """
    Run TreeSHAP on the specified model hash.
    """

    # Load the model
    with gzip.open(f"{MODEL_DIR}/{hash}.pkl.gz", "rb") as f:
        model = pickle.load(f)

    data = pd.read_feather(f"{PREDICTIONS_DIR}/{hash}.feather")
    binding_input = data[[col for col in data.columns if col.endswith("_binding")]]

    assert list(binding_input.columns) == list(model.column_order_when_fitting), "Column order mismatch between input data and model."
    binding_input = binding_input[model.column_order_when_fitting]
    assert list(binding_input.columns) == list(model.column_order_when_fitting), "Column order mismatch between input data and model."

    explainer = shap.TreeExplainer(
            model, 
            feature_perturbation= 'tree_path_dependent', 
            feature_names=binding_input.columns.tolist()
        )

    # Run TreeSHAP
    if normal_or_interaction == "normal":
        logger.info(f"Retrieving normal SHAP values for data with size: {binding_input.shape}.")

        shap_values = explainer.shap_values(binding_input)

        assert shap_values.shape == binding_input.shape, logger.error(f"SHAP values shape {shap_values.shape} does not match input data shape {binding_input.shape}.")
        assert shap_values.shape[0] == data.shape[0], logger.error("Number of rows in SHAP values does not match the input data.")
        logger.success("SHAP values retrieved")

        # Convert SHAP values to a DataFrame and add suffix "_shap" to column names
        shap_df = pd.DataFrame(shap_values, columns=[f"{col.replace('_binding', '_shap')}" for col in binding_input.columns])
        
        # Concatenate SHAP values to the original data
        result = pd.concat([data, shap_df], axis=1)
        # Assert that there are no missing values in the entire result dataframe
        assert not result.isnull().values.any(), "Result dataframe contains missing values."

        logger.success("SHAP values concatenated to the original data by merging horizontally.")

    elif normal_or_interaction == "interaction":
        return NotImplementedError("Interaction SHAP is not implemented yet.")

    if normal_or_interaction == "normal":
        prefix = f"{SHAP_DIR}/regular/normal"
    elif normal_or_interaction == "interaction":
        prefix = f"{SHAP_DIR}/interactions/shap_package/interaction"

    # Save the result to a feather file
    result.to_feather(f"{prefix}_{hash}.feather", compression="lz4")
    logger.success(f"SHAP CALCULATION COMPLETED: {prefix}_{hash}.feather")


if __name__ == "__main__":

    # Configure loguru to log everything to stdout
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run TreeSHAP (either interactions or regular) on all data for XGBRegressor models.")
    parser.add_argument('--run_normal_SHAP', action='store_true', help="Run TreeSHAP on all data (the normal way).")
    parser.add_argument('--run_interaction_SHAP', action='store_true', help="Run TreeSHAP but with feature-feature interactions included.")
    parser.add_argument('--hash', type=str, help="Hash of the model to run SHAP on.")

    parser.add_argument('--run_all_normal', action='store_true', help="Run normal SHAP for all possible hashes.")
    parser.add_argument('--run_all_interaction', action='store_true', help="Run interaction SHAP for all possible hashes.")

    # Parse arguments
    args = parser.parse_args() 

    # Check if the user provided a hash
    if args.hash:
        assert args.run_normal_SHAP or args.run_interaction_SHAP, "Please specify either --run_normal_SHAP or --run_interaction_SHAP."
        
        if args.run_normal_SHAP: 
            main(args.hash, "normal")
        elif args.run_interaction_SHAP:
            main(args.hash, "interaction")
    
    else: 
        assert args.run_all_normal or args.run_all_interaction, "Please specify either --run_all_normal or --run_all_interaction."
        
        model_hashes = [
                    f.split('/')[-1].split('.pkl.gz')[0] for f in glob.glob(f"{MODEL_DIR}/*.pkl.gz")
                ]
        
        if args.run_all_normal: 
            flag = "--run_normal_SHAP"
            SHAP_TYPE_DIR = f"{SHAP_DIR}/regular/"
        elif args.run_all_interaction:
            flag = "--run_interaction_SHAP"
            SHAP_TYPE_DIR = f"{SHAP_DIR}/interactions/shap_package/"

        for model_hash in model_hashes:
            
            shap_type_hash_file = glob.glob(f"{SHAP_TYPE_DIR}/*{model_hash}*.feather")
            assert len(shap_type_hash_file) < 2

            if len(shap_type_hash_file) == 0:
                job_prefix = f"SHAP_{model_hash}"

                os.system(
                    f"sbatch --job-name={job_prefix} -N2 -n{CPUS} --mem={MEM}GB --partition={PARTITION} --account={ACCOUNT} --output={SLURM_DIR}/{job_prefix}.out --error={SLURM_DIR}/{job_prefix}.err --wrap='/bin/python3.11 {__file__} --hash {model_hash} {flag}'"
                )