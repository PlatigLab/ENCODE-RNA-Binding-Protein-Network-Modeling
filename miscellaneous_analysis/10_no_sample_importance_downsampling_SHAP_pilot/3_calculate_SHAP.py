import argparse, gc, os, glob, subprocess

import pandas as pd, polars as pl, numpy as np
import shap, pickle, xgboost
from sklearn.metrics import r2_score


INPUT_DATA_PATH = "/project/PlatigLab/users/yogi/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/02_input_binding_exploration/__featherv2-cache__/"
cell_lines = ["K562", "HepG2"]

def main(cell_line): 

    # Load data indices 
    with open(f"./2_{cell_line}_unique_ids.pkl", "rb") as f:
        unique_ids = pickle.load(f)

    all_ids = unique_ids["train"] | unique_ids["val"] | unique_ids["test"]
    print(f"Processing {len(all_ids)} samples for cell line {cell_line}")
    
    input_data_file = f"{INPUT_DATA_PATH}/{cell_line}_100.feather"
    # Load input data
    df = (
        pl.scan_ipc(input_data_file)
        .filter(pl.col("index").is_in(all_ids))
        .with_columns(pl.col("^*_binding$").cast(pl.UInt8))
        .collect()
        .to_pandas()
    )
    assert len(df) == len(all_ids), f"Dataframe length {len(df)} does not match number of unique ids {len(all_ids)}"

    feature_columns = unique_ids["feature_order"].tolist()
    other_cols = [c for c in df.columns if c not in feature_columns]
    
    df = df[feature_columns + other_cols]
    X = df[feature_columns].copy(deep=True)

    with open(f"./1_{cell_line}.pkl", "rb") as f:
        model = pickle.load(f)
    
    preds = model.predict(X)

    df["Predictions"] = preds

    # Assign partition labels by checking each row's index against the three sets
    train_set = unique_ids["train"]
    val_set = unique_ids["val"]
    test_set = unique_ids["test"]

    partitions = []
    for idx in df["index"]:
        if idx in train_set:
            partitions.append("Train")
        elif idx in val_set:
            partitions.append("Validate")
        elif idx in test_set:
            partitions.append("Test")
        else:
            raise ValueError(f"Unassigned index found: {idx}")

    df["Partition"] = partitions

    r2_results = {}
    for part in ["Train", "Validate", "Test"]:
        sub = df[df["Partition"] == part]
        y_true = sub["Target_PSI"].values
        y_pred = sub["Predictions"].values

        r2 = r2_score(y_true, y_pred)
        print(f"{part} r2_score: {r2:.6f}")

    # Build background_data: subset to feature columns and keep rows with unique combinations of all "_binding" cols
    binding_cols = [c for c in df.columns if c.endswith("_binding")]

    background_data = df.loc[df["Partition"].isin(["Train", "Validate"])].copy(deep=True)
    background_data = background_data.loc[~background_data[binding_cols].duplicated(keep="first")].reset_index(drop=True)
    background_data = background_data[feature_columns]

    print(background_data)
    print(f"Using {len(background_data)} background rows (unique combinations of {len(binding_cols)} binding columns).")

    explainer = shap.TreeExplainer(
        model, 
        data = background_data, 
        model_output="probability",
        feature_perturbation="interventional",
    )

    # Keep the first row for each unique binding combination, then take the feature columns
    shap_input_data = df.loc[
        ~df[binding_cols].duplicated(keep="first"),
        feature_columns,
    ].reset_index(drop=True).copy(deep=True)

    print(f"Calculating SHAP values for {len(shap_input_data)} samples...")
    print(shap_input_data)

    preds = model.predict(shap_input_data)
    shap_data = explainer.shap_values(shap_input_data)

    diff_model_pred = preds - (shap_data.sum(axis=1) + explainer.expected_value)
    print(f"Max prediction difference: {abs(diff_model_pred).max():.5g}")
    print(f"Mean prediction difference: {abs(diff_model_pred).mean():.5g}")

    assert shap_data.shape == shap_input_data.shape, f"SHAP data shape {shap_data.shape} does not match input data shape {shap_input_data.shape}"
    new_shap_cols = [f'{col.replace("_binding", "_shap")}' for col in shap_input_data.columns]

    # Build a DataFrame for SHAP values and concatenate side-by-side with the input features
    shap_df = pd.DataFrame(shap_data, columns=new_shap_cols, index=shap_input_data.index)
    # Combine feature values and SHAP values into a wide table
    shap_wide = pd.concat([shap_input_data.reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)

    # Assert there are no infinities (positive or negative) and no nulls in the combined table
    assert not np.isinf(shap_wide.values).any(), "Infinity values found in the wide SHAP table"
    assert not shap_wide.isnull().values.any(), "Missing/null values found in the wide SHAP table"

    # Sanity checks
    assert shap_wide.shape[1] == shap_input_data.shape[1] + shap_df.shape[1], "Unexpected column count in wide SHAP table"
    assert shap_wide.shape[0] == shap_input_data.shape[0], "Unexpected row count in wide SHAP table"
    print(shap_wide)

    # Ensure join keys exist in shap_wide
    missing_keys = [k for k in binding_cols if k not in shap_wide.columns]
    if missing_keys:
        raise KeyError(f"Missing join keys in shap_wide: {missing_keys}")

    orig_row_count = df.shape[0]
    orig_col_count = df.shape[1]

    print(df.dtypes)
    print(shap_wide.dtypes)

    del X, partitions, preds, background_data, shap_input_data, shap_data, shap_df, diff_model_pred
    gc.collect()

    # Convert to polars in-place and perform a polars join on the binding columns (many-to-one)
    df = pl.from_pandas(df)
    shap_wide = pl.from_pandas(shap_wide)
    
    # Use a left join to preserve original row order/count
    df = df.join(shap_wide, on=binding_cols, how="left", validate = "m:1")

    # Assertions using polars properties
    assert df.height == orig_row_count, f"Row count changed after join: {df.height} != {orig_row_count}"
    assert df.width == orig_col_count + len(new_shap_cols), (
        f"Column count after join is unexpected: {df.width} != {orig_col_count} + {len(new_shap_cols)}"
    )

    # Ensure there are no nulls in the entire dataframe
    assert df.null_count().sum_horizontal().item() == 0, "Null values found in the dataframe after join"

    print(f"Successfully joined SHAP columns: added {len(new_shap_cols)} columns. Dataframe shape is now {(df.height, df.width)}.")
    # Print the resulting polars dataframe
    print(df)

    df.write_ipc(f"./3_{cell_line}_SHAP.feather", compression="lz4")
    
    with open(f"./3_{cell_line}_SHAP_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)

    print(f"Saved SHAP feather file and SHAP explainer for cell line {cell_line}.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Process all cell lines.")
    parser.add_argument("--cell_line", type=str, help="Specify a single cell line to process.")

    args = parser.parse_args()

    if args.all:

        for cell_line in cell_lines:
            cmd = [
                "sbatch",
                "--partition=parallel",
                "--account=platiglab",
                "-N2",
                "-n32",
                "--mem=256GB",
                f"--output=SLURM_{cell_line}.out",
                f"--error=SLURM_{cell_line}.err",
                f"--wrap=python3.11 ./3_calculate_SHAP.py --cell_line {cell_line}",
            ]
            subprocess.run(cmd, check=True)

    elif args.cell_line:

        main(args.cell_line)

    else:
        print("Please specify --all or --cell_line.")