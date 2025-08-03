import glob, argparse, subprocess, os
import polars as pl
from loguru import logger


SHAP_NORMAL_DIR = "../outputs/SHAP/regular/normal/shap_values/"
SHAP_INTERACTION_DIR = "../outputs/SHAP/interactions/shap_package/"

ORIGINAL_DATA_DIR = "../outputs/predictions/XGBRegressor/"

CPUS = 32
MEM= 256
PARTITION="parallel"
ACCOUNT="platiglab"


def main(mode, shap_file): 
    assert mode == 'normal', "Only 'normal' mode is implemented in this script."
    assert shap_file is not None, "SHAP file must be provided in normal mode."

    # Lazy load the SHAP file
    shap_lf = pl.scan_ipc(shap_file)

    # Extract hash from shap_file name
    hash = shap_file.split("/")[-1].split("_")[0]

    # Find the original data file
    original_data_file = glob.glob(f"{ORIGINAL_DATA_DIR}/{hash}*.feather")
    assert len(original_data_file) == 1, f"Expected exactly one original data file for hash {hash}, found {len(original_data_file)}."
    original_data_file = original_data_file[0]

    # Lazy load the original data file
    original_lf = pl.scan_ipc(original_data_file)

    # Get number of columns in SHAP file (lazily)
    shap_schema = shap_lf.collect_schema().names()
    shap_num_cols = len(shap_schema)

    # Get number of rows in original data file (lazily)
    original_num_rows = original_lf.select(pl.len()).collect().item()

    # Get schema and select binding and SHAP columns
    binding_cols = [col for col in shap_schema if col.endswith("_binding")]
    shap_cols = [col for col in shap_schema if col.endswith("_shap")]
    
    selected_cols = binding_cols + shap_cols
    shap_selected_lf = shap_lf.select(selected_cols)

    # Left join original data with SHAP data on binding columns
    joined_lf = original_lf.join(
        shap_selected_lf,
        on=binding_cols,
        how="left", 
        validate='m:1'
    )

    # Collect the joined dataframe
    joined_df = joined_lf.collect()

    # Assert no nulls in joined columns
    assert joined_df.null_count().sum_horizontal().item() == 0, "Null values found in joined SHAP columns."

    # Assert number of columns and rows
    assert joined_df.shape[1] == shap_num_cols, "Number of columns does not match SHAP file."
    assert joined_df.shape[0] == original_num_rows, "Number of rows does not match original data file."

    # Assert that for each unique combination of binding columns, the SHAP columns are the same
    grouped = joined_df.group_by(binding_cols).agg(
        [pl.struct(shap_cols).n_unique().alias("shap_row_nunique")]
    ).select('shap_row_nunique')
    assert grouped["shap_row_nunique"].max() == 1, (
        "For at least one unique combination of binding columns, SHAP columns have inconsistent values."
    )

    # Cast all binding columns to uint8 in a single call
    joined_df = joined_df.with_columns([pl.col(col).cast(pl.UInt8) for col in binding_cols])

    # Save the joined dataframe as a feather file with the pattern {hash}.feather
    output_path = f"{SHAP_NORMAL_DIR if mode == 'normal' else SHAP_INTERACTION_DIR}/{hash}.feather"
    joined_df.write_ipc(output_path, compression="lz4")

    # Remove the original SHAP file
    os.remove(shap_file)
    logger.success(f"Completed! Joined file to {output_path} and removed SHAP file {shap_file}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Join unique binding SHAP results with all of the data.")
    
    parser.add_argument(
        "--mode",
        choices=["normal", "interaction"],
        required=False,
        help="Select processing mode: 'normal' or 'interaction'"
    )
    parser.add_argument('--shap_file', type=str, required=False, help="Path to SHAP file.")
    parser.add_argument('--run_all', choices=["normal", "interaction"], required=False, help="Run all SHAP files in the specified mode.")
    
    args = parser.parse_args()

    if args.mode is not None: 
        assert args.shap_file is not None, "SHAP file must be provided in normal mode."
        assert args.mode !='interaction', NotImplementedError("Interaction mode is not implemented yet.")
        
        main(args.mode, args.shap_file)

    
    elif args.run_all is not None:
        if args.run_all == "normal":
            GLOB_DIR = SHAP_NORMAL_DIR
        elif args.run_all == "interaction":
            GLOB_DIR = SHAP_INTERACTION_DIR
        else:
            raise ValueError("Invalid --run_all option. Choose 'normal' or 'interaction'.")
        
        shap_files = glob.glob(f"{GLOB_DIR}/*_unique_binding.feather")
        assert shap_files, f"No SHAP files found in {GLOB_DIR}."

        for shap_file in shap_files:

            hash = shap_file.split("/")[-1].split("_")[0]

            cmd = [
                "sbatch",
                f"--account={ACCOUNT}",
                f"--partition={PARTITION}",
                f"-N 2" if PARTITION == "parallel" else '1',
                f"-n {CPUS}",
                f"--mem={MEM}G",
                f"--output=../outputs/SLURM_logs/join_SHAP_with_all_data_{hash}.out",
                f"--error=../outputs/SLURM_logs/join_SHAP_with_all_data_{hash}.err",
                f"--job-name={hash}",
                "--wrap",
                (
                    f"python3.11 {__file__} "
                    f"--mode {args.run_all} "
                    f"--shap_file {shap_file}"
                )
            ]
            
            logger.info(f"Submitting job for SHAP file: {shap_file}")
            logger.info(f"Command: {cmd}")

            subprocess.run(cmd)

    else:
        raise ValueError("Please specify a flag.")