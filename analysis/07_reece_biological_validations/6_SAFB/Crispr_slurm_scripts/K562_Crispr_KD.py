import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os, argparse, glob, gzip, pickle, shap, gc
from pathlib import Path
from loguru import logger
import polars.selectors as cs
import polars as pl
_=pl.Config.set_tbl_cols(100000)
_=pl.Config.set_tbl_rows(10000)
_=pl.Config.set_tbl_width_chars(10000)
_=pl.Config.set_fmt_str_lengths(10000)

hash_table_path = "/project/PlatigLab/users/yogi/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/06_first_order_SHAP_analysis/outputs/hash_metadata/hash_metadata.tsv"

# -----------------------
# Load data
# -----------------------

# Load in the BAT
# Function to get path for big table

def get_path(cell_line: str, path_file: str = "data_path.txt") -> str:
    """
    Reads a base directory path from a text file and returns the full path
    to the cell line data directory.

    Args:
        cell_line (str): The name of the cell line (e.g. "K562" or "HepG2").
        path_file (str): Path to the text file containing the base directory path.

    Returns:
        str: Full path to the data file for the given cell line.
    """
    # Read base path from file
    base_path = Path(path_file).read_text().strip()
    
    # Build full path
    full_path = Path(base_path) / f"{cell_line}_all-data.feather"
    
    return str(full_path)


MODEL_DIR = "/project/PlatigLab/users/yogi/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/04_run_final_models_and_SHAP/outputs/pickled_models/XGBRegressor"

K562_hashes = ["kzbv", "mzdv", "niwy", "nuyh", "oodr"]

# -----------------------
# Functions
# -----------------------

# Function to load rMATS, filter for read counts, and join with BAT

def load_rmats(rbp, BAT, cell_line, read_counts_threshold):

    print(f"This is for {rbp} in {cell_line} with read counts = {read_counts_threshold}")

    # Read in rMATS
    rMATS_data = pl.read_csv(
        f'/project/PlatigLab/data/ENCORE2026/rMATS_analysis/v29_rMATS_RBPKD/{rbp}-CRISPR-{cell_line}/SE.MATS.JC.txt',
        separator='\t'
    )

    # Keep select columns
    rMATS_filtered = rMATS_data[
        ["FDR", "geneSymbol", "strand", "chr",
         "exonStart_0base", "exonEnd",
         "upstreamES", "upstreamEE",
         "downstreamES", "downstreamEE", "IJC_SAMPLE_1",
         "IJC_SAMPLE_2", "SJC_SAMPLE_1", "SJC_SAMPLE_2",
         "ID", "IncFormLen", "SkipFormLen",
         "PValue", "IncLevelDifference"]
    ]

    print(f"The shape of the rMATS file for {rbp} in {cell_line} is {rMATS_filtered.shape}")

    # Filter for read counts
    rMATS_filtered = rMATS_filtered.with_columns([
        (
            pl.col("IJC_SAMPLE_1").str.split(",").list.get(0).cast(pl.Int64) + pl.col("SJC_SAMPLE_1").str.split(",").list.get(0).cast(pl.Int64)).alias("Counts_KD-1"),
        (
            pl.col("IJC_SAMPLE_1").str.split(",").list.get(1).cast(pl.Int64) + pl.col("SJC_SAMPLE_1").str.split(",").list.get(1).cast(pl.Int64)).alias("Counts_KD-2"),
        (
            pl.col("IJC_SAMPLE_2").str.split(",").list.get(0).cast(pl.Int64) + pl.col("SJC_SAMPLE_2").str.split(",").list.get(0).cast(pl.Int64)).alias("Counts_CTRL-1"),
        (
            pl.col("IJC_SAMPLE_2").str.split(",").list.get(1).cast(pl.Int64) + pl.col("SJC_SAMPLE_2").str.split(",").list.get(1).cast(pl.Int64)).alias("Counts_CTRL-2"),
    ])


    # Filter for read counts according to what is specified in the function call
    rMATS_filtered = rMATS_filtered.filter(
        (
            (pl.col("Counts_KD-1") > read_counts_threshold) |
            (pl.col("Counts_KD-2") > read_counts_threshold)
        ) &
        (
            (pl.col("Counts_CTRL-1") > read_counts_threshold) |
            (pl.col("Counts_CTRL-2") > read_counts_threshold)
        )
    )

    print(f"The shape of the read-counts filtered rMATS file for {rbp} in {cell_line} is {rMATS_filtered.shape}")

    # Build exon string AND keep IncLevelDifference and FDR
    rMATS_events = rMATS_filtered.select([
        pl.col("ID"),
        pl.col("IncLevelDifference"),
        pl.col("FDR"), # <-- carry these through
        pl.when(pl.col("strand") == "+")
        .then(
            pl.concat_str([
                pl.col("chr"),
                pl.col("strand"),
                pl.col("upstreamES"),
                pl.col("upstreamEE"),
                pl.col("exonStart_0base"),
                pl.col("exonEnd"),
                pl.col("downstreamES"),
                pl.col("downstreamEE")
            ], separator="_")
        )
        .otherwise(
            pl.concat_str([
                pl.col("chr"),
                pl.col("strand"),
                pl.col("downstreamEE"),
                pl.col("downstreamES"),
                pl.col("exonEnd"),
                pl.col("exonStart_0base"),
                pl.col("upstreamEE"),
                pl.col("upstreamES")
            ], separator="_")
        )
        .alias("string")
    ])

    N_ELEMENTS = 8
    DELIMITER = "_"

    BAT_extracted = BAT.with_columns(
        pl.col("index")
        .str.split(by=DELIMITER)
        .list.slice(0, N_ELEMENTS)
        .list.join(DELIMITER)
        .alias("string")
    )

    # JOIN — IncLevelDifference stays matched to string
    RBP_KD_matches_BAT = BAT_extracted.join(
        rMATS_events,
        on="string",
        how="inner"
    ).drop("string")

    # rename column
    RBP_KD_matches_BAT = RBP_KD_matches_BAT.rename(
        {"IncLevelDifference": "Crispr rMATS dPSI"}
    )

    RBP_KD_matches_BAT = RBP_KD_matches_BAT.rename(
        {"FDR": "Crispr rMATS FDR"}
    )

    # drop shap columns
    RBP_KD_matches_BAT = RBP_KD_matches_BAT.drop(
        pl.col("^.*_shap.*$")
    )

    print(f"Here is the shape of matches between {rbp} KD events and the BAT for {cell_line}:")
    print(RBP_KD_matches_BAT.shape)

    # Get just control rows
    CTRL_rows = RBP_KD_matches_BAT.filter(pl.col("index").str.contains("CTRL"))

    # Get only unique locations to get the binding pattern for that location

    unique_ctrl_rows = (
        CTRL_rows.with_columns(
            pl.col("index")
            .str.split(by=DELIMITER)
            .list.slice(0, N_ELEMENTS)  # Slice the first N elements of the resulting list
            .alias("prefix")
        )
        .unique(subset="prefix")
        .drop("prefix")
    )

    print("Shape of unique ctrl rows")
    print(unique_ctrl_rows.shape)

    duplicated = pl.concat([
        unique_ctrl_rows.with_columns(pl.lit("CTRL").alias("Row Type")),
        unique_ctrl_rows.with_columns(pl.lit("IS-KD").alias("Row Type"))
    ])

    print("Shape of duplicated:")
    print(duplicated.shape)

    # Do the in-silico KD
    
    cols = [f"{rbp}_{i}_binding" for i in range(1, 7)]

    IS_KD_df = duplicated.with_columns([
        pl.when(pl.col("Row Type") == "IS-KD")
        .then(0)
        .otherwise(pl.col(c))
        .alias(c)
        for c in cols
    ])

    # Keep only binding cols, index, and row type, and dPSI

    IS_KD_df = IS_KD_df.select(
        "index",
        "Row Type",
        "Crispr rMATS dPSI",
        "Crispr rMATS FDR",
        cs.ends_with("_binding")
    )

    final = (
        IS_KD_df
        .with_columns([
            (pl.col("Crispr rMATS dPSI") * -1).alias("Crispr rMATS dPSI"),
            pl.lit(cell_line).alias("cell_line"),
            pl.lit(rbp).alias("BP for")
        ])
        .select(
            "cell_line",
            "BP for",
            pl.exclude("cell_line", "BP for")
        )
    )
    
    print(f"The shape of final df for {rbp} in {cell_line} is {final.shape}")
    
    return final


# Function to run TreeSHAP for each hash and return average SHAP values


def get_shap(hashes, binding_pattern):
    """
    Run TreeSHAP for each hash and return average SHAP values
    """

    cell_line = binding_pattern.select(pl.col("cell_line").first()).item()
    bp_for = binding_pattern.select(pl.col("BP for").first()).item()
    print(f"DF for {bp_for} {cell_line}")

    binding_cols = [col for col in binding_pattern.collect_schema().names() if col.endswith("_binding")]

    # Select all data for SHAP analysis (same for all hashes)
    all_data = (
        binding_pattern
        .select(binding_cols)
        .to_pandas()
    )


    shap_stack = []

    for hash in hashes:

        print(hash)

        # Load model
        with gzip.open(f"{MODEL_DIR}/{hash}.pkl.gz", 'rb') as f:
            model = pickle.load(f)

        explainer = shap.TreeExplainer(
            model,
            model_output="raw",
            feature_perturbation="tree_path_dependent",
        )

        assert all_data.columns.tolist() == list(model.column_order_when_fitting)

        logger.info(f"{hash} → Retrieving SHAP values for data with shape: {all_data.shape}")

        shap_values = explainer.shap_values(
            all_data,
            approximate=False,
            check_additivity=True,
        )

        shap_stack.append(shap_values)


    # stack: (n_hash, n_rows, n_features)
    shap_stack = np.stack(shap_stack, axis=0)

    # average across hashes
    shap_mean = np.mean(shap_stack, axis=0)


    # convert averaged SHAP → polars
    shap_df = pl.DataFrame(
        shap_mean,
        schema=[col.replace("_binding", "_shap") for col in binding_cols]
    )


    # combine binding + shap
    result = pl.concat([binding_pattern, shap_df], how="horizontal")

    return result


# Function to make table comparing SHAP values for CTRL vs IS-KD for each position, along with dPSI and FDR from rMATS

def make_IS_KD_table(custom_crispr_bat: pl.DataFrame, rbp: str):

    cell_line = custom_crispr_bat.select(pl.col("cell_line").first()).item()
    bp_for = custom_crispr_bat.select(pl.col("BP for").first()).item()
    print(f"DF for {bp_for} {cell_line}")

    dfs = []

    for position in range(1, 7):

        binding_col = f"{rbp}_{position}_binding"
        shap_col = f"{rbp}_{position}_shap"

        ctrl = (
            custom_crispr_bat
            .filter(
                (pl.col("Row Type") == "CTRL") &
                (pl.col(binding_col) == 1)
            )
            .select([
                "index",
                shap_col,
                "Crispr rMATS dPSI",
                "Crispr rMATS FDR"
            ])
            .rename({
                shap_col: "CTRL_SHAP",
                "Crispr rMATS dPSI": "dPSI",
                "Crispr rMATS FDR": "FDR"
            })
        )

        kd = (
            custom_crispr_bat
            .filter(pl.col("Row Type") == "IS-KD")
            .select([
                "index",
                shap_col
            ])
            .rename({shap_col: "KD_SHAP"})
        )

        out = (
            ctrl
            .join(kd, on="index", how="inner")
            .with_columns(
                (pl.col("CTRL_SHAP") - pl.col("KD_SHAP"))
                .alias("CTRL - KD Local SHAP")
            )
            .with_columns([
                pl.lit(binding_col).alias("Feature"),
                pl.lit(rbp).alias("RBP-IS-KD-Target"),
                pl.lit(position).alias("Position"),
                pl.lit(cell_line).alias("cell_line"), 
            ])
            .select([
                "Feature",
                "RBP-IS-KD-Target",
                "Position",
                "cell_line",
                "index",
                "dPSI",
                "CTRL - KD Local SHAP",
                "FDR",
            ])
        )

        dfs.append(out)

    return pl.concat(dfs)

# Function to append results to a csv (or create if it doesn't exist)

def append_result(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    file_exists = os.path.exists(path)

    with open(path, "a") as f:
        df.write_csv(
            f,
            include_header=not file_exists
        )
# -----------------------
# pipeline (ONE RBP)
# -----------------------

def run_pipeline_single(
    rbp,
    BAT,
    cell_line,
    read_counts_threshold,
    hashes
):

    binding_pattern = load_rmats(rbp, BAT, cell_line, read_counts_threshold)

    print("one", flush=True)

    custom_crispr_bat = get_shap(hashes, binding_pattern)

    print("two", flush=True)

    final_table = make_IS_KD_table(custom_crispr_bat, rbp)

    print("three", flush=True)

    append_result(final_table, path=f"/project/PlatigLab/users/reece/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/07_reece_biological_validations/6_SAFB/3_results_{rbp}.csv")

    print(f"Finished run for {rbp} in {cell_line}")


# -----------------------
# main
# -----------------------

if __name__ == "__main__":

    rbp = sys.argv[1]

    precomputed_path = f"precomputed_rmats/{rbp}_K562.feather"

    binding_pattern = pl.read_ipc(precomputed_path)   # tiny — only the matched rows

    cell_line = "K562"
    hashes = K562_hashes

    custom_crispr_bat = get_shap(hashes, binding_pattern)
    final_table = make_IS_KD_table(custom_crispr_bat, rbp)
    append_result(
        final_table,
        path=f"/project/PlatigLab/users/reece/ENCODE-RNA-Binding-Protein-Network-Modeling/"
             f"analysis/07_reece_biological_validations/6_SAFB/3_results_{rbp}.csv"
    )
    print(f"Finished run for {rbp} in K562")