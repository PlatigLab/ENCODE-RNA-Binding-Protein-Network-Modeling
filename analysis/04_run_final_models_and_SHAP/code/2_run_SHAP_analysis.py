import sys, os, argparse, glob, gzip, pickle, shap, gc
import pandas as pd, polars as pl, numpy as np
from loguru import logger

MODEL_DIR = "../outputs/pickled_models/XGBRegressor/"
PREDICTIONS_DIR = "../outputs/predictions/XGBRegressor/"
SHAP_DIR = "../outputs/SHAP/"
SLURM_DIR="../outputs/SLURM_logs/"

CPUS = 16
MEM= 256
PARTITION="standard"
ACCOUNT="platiglab"


def main(hash, normal_or_interaction):
    """
    Run TreeSHAP on the specified model hash.
    """

    assert normal_or_interaction == 'normal', "Only 'normal' SHAP is supported in this script."
    
    # Load model
    with gzip.open(f"{MODEL_DIR}/{hash}.pkl.gz", 'rb') as f:
        model = pickle.load(f)

    predictions_lf = pl.scan_ipc(f"{PREDICTIONS_DIR}/{hash}.feather").sort('index')
    binding_cols = [col for col in predictions_lf.collect_schema().names() if col.endswith("_binding")]

    explainer = shap.TreeExplainer(
        model, 
        model_output="raw",
        feature_perturbation="tree_path_dependent",
    )

    # Select all data for SHAP analysis
    all_data = (
        predictions_lf
        .unique(subset=binding_cols, keep='first', maintain_order=True)
        .select(binding_cols)
        .collect()
        .to_pandas()
    )

    assert all_data.columns.tolist() == list(model.column_order_when_fitting)
    logger.info(f"Retrieving SHAP values for data with shape: {all_data.shape}")

    # Run SHAP analysis
    shap_values = explainer.shap_values(
        all_data, 
        approximate=False,
        check_additivity=True,
    )

    # Method 1: Use the pickled model's .predict functionality on all_data
    model_preds = model.predict(all_data, output_margin=True)
    del all_data  # Free memory
    gc.collect()

    assert model_preds.shape[0] == shap_values.shape[0], f"Model predictions shape ({model_preds.shape}) does not match SHAP values shape ({shap_values.shape})"
    diff_model_pred = model_preds - (shap_values.sum(axis=1) + explainer.expected_value)

    logger.info(f"[Model.predict] Absolute mean difference: {abs(diff_model_pred).mean()}")
    logger.info(f"[Model.predict] Max absolute difference: {abs(diff_model_pred).max()}")

    # Method 2: Compute the difference between predictions and the sum of SHAP values plus expected value (using predictions column)
    predictions_np = (
        predictions_lf
        .unique(subset=binding_cols, keep='first', maintain_order=True)
        .select("Predictions")
        .collect()["Predictions"]
        .to_numpy()
    )
    predictions_np = np.log(predictions_np / (1 - predictions_np))

    assert predictions_np.shape[0] == shap_values.shape[0], f"Predictions shape ({predictions_np.shape}) does not match SHAP values shape ({shap_values.shape})"
    diff_pred_col = predictions_np - (shap_values.sum(axis=1) + explainer.expected_value)

    logger.info(f"[Predictions column] Absolute mean difference: {abs(diff_pred_col).mean()}")
    logger.info(f"[Predictions column] Max absolute difference: {abs(diff_pred_col).max()}")

    logger.info(f"Expected value: {explainer.expected_value}")

    # get unique predictions DataFrame with binding columns
    predictions_pl = predictions_lf.unique(subset=binding_cols, keep='first', maintain_order=True).collect()

    # Create a polars DataFrame for SHAP values with appropriate column names
    shap_values = pl.DataFrame(shap_values, schema=[col.replace("_binding", "_shap") for col in binding_cols])
    
    # Ensure the shape of shap_values matches the number of rows and features
    assert shap_values.height == predictions_pl.height, f"SHAP values height ({shap_values.height}) does not match predictions height ({predictions_pl.height})"
    assert shap_values.width == len(binding_cols), f"SHAP values width ({shap_values.width}) does not match number of binding columns ({len(binding_cols)})"

    # Concatenate along columns
    shap_values = pl.concat([predictions_pl, shap_values], how="horizontal")

    # Final assertion to check shape
    expected_cols = predictions_pl.width + len(binding_cols)
    assert shap_values.height == predictions_pl.height, f"SHAP values height ({shap_values.height}) != predictions_pl height ({predictions_pl.height})"
    assert shap_values.width == expected_cols, f"Resulting DataFrame should have {expected_cols} columns, got {shap_values.width}"
    
    del predictions_pl  # Free memory
    gc.collect()

    # Check for null or missing values and log a warning if any are found
    null_count = shap_values.null_count().sum_horizontal().item()
    nan_count = shap_values.select(pl.selectors.float().is_nan().sum()).sum_horizontal().item()

    assert null_count == 0, f"Found {null_count} null values in the SHAP values DataFrame."
    assert nan_count == 0, f"Found {nan_count} NaN values in the SHAP values DataFrame."

    shap_values.write_ipc(f"{SHAP_DIR}/regular/normal/shap_values/{hash}_unique_binding.feather", compression="lz4")
    logger.success(f"SHAP value dataframe with shape {shap_values.shape} saved to {SHAP_DIR}/regular/normal/shap_values/{hash}_unique_binding.feather")

    with open(f"{SHAP_DIR}/regular/normal/explainer_objects/{hash}.pkl", "wb") as f:
        pickle.dump(explainer, f)

    logger.success(f"Explainer object saved to {SHAP_DIR}/regular/normal/explainer_objects/{hash}.pkl")
    logger.success(f"COMPLETED: SHAP analysis for model hash {hash} with SHAP type '{normal_or_interaction}'")


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
        
        model_hashes = sorted([
                    f.split('/')[-1].split('.pkl.gz')[0] for f in glob.glob(f"{MODEL_DIR}/*.pkl.gz")
                ])
        
        if args.run_all_normal: 
            flag = "--run_normal_SHAP"
            SHAP_TYPE_DIR = f"{SHAP_DIR}/regular/"
        elif args.run_all_interaction:
            flag = "--run_interaction_SHAP"
            SHAP_TYPE_DIR = f"{SHAP_DIR}/interactions/shap_package/"

        for model_hash in model_hashes:
            
            shap_type_hash_file = glob.glob(f"{SHAP_TYPE_DIR}/**/*{model_hash}*.feather", recursive=True)
            assert len(shap_type_hash_file) <= 1

            if len(shap_type_hash_file) == 0:
                job_prefix = f"SHAP_{model_hash}"

                os.system(
                    f"sbatch --job-name={job_prefix} -n{CPUS} --mem={MEM}GB --partition={PARTITION} --account={ACCOUNT} --output={SLURM_DIR}/{job_prefix}.out --error={SLURM_DIR}/{job_prefix}.err --wrap='/bin/python3.11 {__file__} --hash {model_hash} {flag}'"
                )