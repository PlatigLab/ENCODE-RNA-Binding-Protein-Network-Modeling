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

# -----------------------
# Load data
# -----------------------

# Load in the BAT
# Function to get path for big table

def get_path(cell_line: str, path_file: str = "/project/PlatigLab/users/reece/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/07_reece_biological_validations/data_path.txt") -> str:
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

# -----------------------
# Functions
# -----------------------

# Function to load rMATS, filter for read counts, and join with BAT

def load_rmats(rbp, BAT, cell_line, read_counts_threshold):

    print(f"This is for {rbp} in {cell_line} with read counts = {read_counts_threshold}")

    # Read in rMATS
    rMATS_data = pl.read_csv(
        f'/project/PlatigLab/data/ENCORE2026/XGB_SHAP_CRISPR/rMATS_run/CRISPR_KD_rMATS/{rbp}-CRISPR-{cell_line}/SE.MATS.JC.txt',
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
            .alias("location")
        )
        .unique(subset="location")
        .drop("location")
    )

    print("Shape of unique ctrl rows")
    print(unique_ctrl_rows.shape)

    # Keep only binding cols, index, and row type, and dPSI

    ctrl_crispr_bat = unique_ctrl_rows.select(
        "index",
        "Crispr rMATS dPSI",
        "Crispr rMATS FDR",
        cs.ends_with("_binding"),
        cs.ends_with("_shap")
    )

    final = (
        ctrl_crispr_bat
        .with_columns([
            (pl.col("Crispr rMATS dPSI") * -1).alias("Crispr rMATS dPSI"),
            pl.lit(cell_line).alias("cell_line"),
            pl.lit(rbp).alias("rMATS RBP")
        ])
        .select(
            "cell_line",
            "rMATS RBP",
            pl.exclude("cell_line", "rMATS RBP")
        )
    )
    
    print(f"The shape of final df for {rbp} in {cell_line} is {final.shape}")
    
    return final


# Function to make table with CTRL SHAP, along with dPSI and FDR from rMATS


def make_table(custom_crispr_bat: pl.DataFrame, rbp: str):

    cell_line = custom_crispr_bat.select(pl.col("cell_line").first()).item()
    bp_for = custom_crispr_bat.select(pl.col("rMATS RBP").first()).item()
    print(f"DF for {bp_for} {cell_line}")

    dfs = []

    for position in range(1, 7):
        
        binding_col = f"{rbp}_{position}_binding"
        shap_col = f"{rbp}_{position}_shap"
    
        out = (
            custom_crispr_bat
            .filter(pl.col(binding_col) == 1)
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
                "CTRL_SHAP",
                "FDR",
            ])
        )
        dfs.append(out)

    return pl.concat(dfs)

# Function to append results to a csv (or create if it doesn't exist)

def write_result(df, path):
    """
    Write result to CSV, overwriting if it exists.
    This makes the pipeline idempotent (safe to rerun).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df.write_csv(path)
# -----------------------
# pipeline (ONE RBP)
# -----------------------

def run_pipeline_single(
    rbp,
    BAT,
    cell_line,
    read_counts_threshold,
):

    binding_pattern = load_rmats(rbp, BAT, cell_line, read_counts_threshold)

    print("one", flush=True)
    
    final_table = make_table(binding_pattern, rbp)

    print("two", flush=True)

    write_result(final_table, path=f"/project/PlatigLab/users/reece/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/07_reece_biological_validations/1_for_figures/4_Crispr_CTRL_SHAP/K562_results_{rbp}.csv")

    print(f"Finished run for {rbp} in {cell_line}")


# -----------------------
# main
# -----------------------

if __name__ == "__main__":

    import sys

if __name__ == "__main__":
    
    rbp = sys.argv[1]

    precomputed_path = f"precomputed_rmats_K562/{rbp}_K562.feather"

    binding_pattern = pl.read_ipc(precomputed_path)   # tiny — only the matched rows

    cell_line = "K562"

    final_table = make_table(binding_pattern, rbp)

    write_result(
        final_table,
        path=f"/project/PlatigLab/users/reece/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/07_reece_biological_validations/1_for_figures/4_Crispr_CTRL_SHAP/K562_results_{rbp}.csv")
    
    print(f"Finished run for {rbp} in K562")