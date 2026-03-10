import gzip, pickle, yaml, argparse, pathlib, os, math
import shap, polars as pl, numpy as np



def main(cell_line=None, unique_binding_pattern_ID_range=None):
    
    with open("1_metadata_variables.yaml", 'r') as f:
        metadata_variables = yaml.safe_load(f)

    assert cell_line is not None and unique_binding_pattern_ID_range is not None, "Both cell_line and unique_binding_pattern_ID_range must be provided."
    
    split = unique_binding_pattern_ID_range.split("-")
    assert len(split) == 2, "unique_binding_pattern_ID_range must be in the format 'start-end'."
    
    start_id, end_id = int(split[0]), int(split[1])
    assert end_id > start_id, "end_id must be greater than start_id."

    if (end_id - start_id + 1) != metadata_variables["SLURM_JOB_BATCH_SIZE"]:
        
        total_ubp_ids_path = f"{metadata_variables['UBP_ID_TABLE_DIR']}/{cell_line}_unique_binding_pattern_ID_reference_table.tsv.gz"
        max_ubp_id = pl.scan_csv(
            total_ubp_ids_path,
            separator="\t", 
        ).select("Unique Binding Pattern ID #").max().collect().item()

        assert end_id == max_ubp_id, (
            f"When the range size is not equal to SLURM_JOB_BATCH_SIZE ({metadata_variables['SLURM_JOB_BATCH_SIZE']}), "
            f"the end_id must be the maximum UBP ID ({max_ubp_id}) for the cell line {cell_line}."
        )

    OUTPUT_FILE = f"{metadata_variables['SHAP_INTERACTION_VALUES_DIR']}/{cell_line}_UBP_IDs_{start_id}-{end_id}_shap_interactions.feather"
    assert not pathlib.Path(OUTPUT_FILE).exists(), f"Output file {OUTPUT_FILE} already exists. Please remove it before running the script."

    ubp_id_table_path = f"{metadata_variables['UBP_ID_TABLE_DIR']}/{cell_line}_unique_binding_pattern_ID_reference_table.tsv.gz"
    ubp_id_table = pl.scan_csv(
        ubp_id_table_path,
        separator="\t", 
    ).filter(
        (pl.col("Unique Binding Pattern ID #") >= start_id) & 
        (pl.col("Unique Binding Pattern ID #") <= end_id)
    ).collect()
    
    unique_ids = ubp_id_table["Unique Binding Pattern ID #"].to_list()
    assert unique_ids == list(range(start_id, end_id + 1)), (
        f"UBP ID table does not contain the expected unique binding pattern IDs from {start_id} to {end_id}."
    )
    
    ubp_id_table = ubp_id_table.drop("Unique Binding Pattern ID #")
    assert all(col.endswith("_binding") for col in ubp_id_table.columns), "All columns in ubp_id_table must end with '_binding'."
    ubp_id_table = ubp_id_table.with_columns([pl.col(col).cast(pl.UInt8) for col in ubp_id_table.columns])
    
    expected_rows = end_id - start_id + 1

    assert ubp_id_table.height == expected_rows, (
        f"Expected {expected_rows} rows in UBP ID table for IDs {start_id}-{end_id}, but got {ubp_id_table.height} rows."
    )
    assert ubp_id_table.unique(ubp_id_table.columns).height == expected_rows, (
        "UBP ID table contains duplicate rows."
    )

    model_hash= metadata_variables["XGBOOST_BEST_MODEL_HASHES"][cell_line]
    model_path = f'{metadata_variables["MODEL_DIR"]}/{model_hash}.pkl.gz'

    # Load the model
    with gzip.open(model_path, 'rb') as f:
        model = pickle.load(f)

    ubp_id_table = ubp_id_table.to_pandas()

    assert list(ubp_id_table.columns) == list(model.column_order_when_fitting), "Columns in explanation rows do not match model's expected input."
    log_odds_predictions = model.predict(ubp_id_table, output_margin=True).tolist()

    # Create SHAP explainer
    explainer = shap.TreeExplainer(
        model, 
        feature_perturbation = "tree_path_dependent", 
        model_output = "raw", 
    )

    all_rows_shap_interaction_values = explainer.shap_interaction_values(
        ubp_id_table, 
    )

    assert isinstance(explainer.expected_value, (float, np.floating)), (
        f"Expected explainer.expected_value to be a float, got {type(explainer.expected_value)}: {explainer.expected_value}"
    )
    expected_value = explainer.expected_value

    with open(f'{metadata_variables["EXPECTED_VALUES_DIR"]}/{cell_line}_{start_id}-{end_id}_expected_value.txt', 'w') as f:
        f.write(f"{expected_value}")

    # build dictionary of feature interaction names and values being tuple of the indices in the interaction matrix
    num_features = len(ubp_id_table.columns)
    columns = list(ubp_id_table.columns)
    
    feature_interaction_dict = {}
    for i in range(num_features):
        feature_a = columns[i]
        
        # Main effect (diagonal)
        feature_interaction_name = f"{feature_a}-main-shap".replace("_binding", '')
        feature_interaction_dict[feature_interaction_name] = (i, i)
        
        # Unique pairs (combinations, not permutations)
        for j in range(i + 1, num_features):
            feature_b = columns[j]
            feature_interaction_name = f"{feature_a}-{feature_b}-interaction-shap".replace("_binding", '')
            feature_interaction_dict[feature_interaction_name] = (i, j)

    assert len(feature_interaction_dict) == math.comb(num_features, 2) + num_features, (
        f"Expected {math.comb(num_features, 2) + num_features} feature interactions, but got {len(feature_interaction_dict)}."
    )

    # Prepare feature names in order for columns
    all_rows = []
    for i, explanation in enumerate(all_rows_shap_interaction_values):

        shap_diff = log_odds_predictions[i] - expected_value
        shap_sum = explanation.sum()
        assert abs(shap_diff - shap_sum) < metadata_variables["PRECISION_THRESHOLD"], (
            f"SHAP interaction values do not sum to prediction difference for row {i}. "
            f"Difference: {shap_diff}, Sum: {shap_sum}, Abs Difference: {abs(shap_diff - shap_sum)}"
        )

        for a in range(explanation.shape[0]):
            for b in range(explanation.shape[0]): 
                assert abs(explanation[a][b] - explanation[b][a]) < metadata_variables["PRECISION_THRESHOLD"], f"SHAP interaction values are not symmetric at ({a}, {b}). {explanation[a][b]} != {explanation[b][a]}"

        # Build the row as a numpy array
        row_values = []
        for feature_name in feature_interaction_dict.keys():
            assert len(feature_interaction_dict[feature_name]) == 2, "Each feature interaction must map to a tuple of two indices."
            a, b = feature_interaction_dict[feature_name]
            if a != b:
                value = explanation[a][b] + explanation[b][a]
            else:
                value = explanation[a][b]
            row_values.append(value)

        all_rows.append(row_values)

    # After the loop, create the polars DataFrame from numpy arrays (without the ID column)
    all_rows = np.stack(all_rows)    
    all_rows = pl.DataFrame(
        all_rows,
        schema=list(feature_interaction_dict.keys()), 
        orient="row"
    )

    # Add the unique binding pattern ID column as integer
    all_rows = all_rows.with_columns([
        pl.Series("Unique Binding Pattern ID #", np.array(unique_ids, dtype=np.uint32))
    ])

    # Move the ID column to the front
    all_rows = all_rows.select(["Unique Binding Pattern ID #"] + list(feature_interaction_dict.keys()))
    # Assert there are no null or missing values
    assert all_rows.null_count().sum_horizontal().item() == 0, "There are null or missing values in the result DataFrame."
    assert all_rows.height == expected_rows, (
        f"Expected {expected_rows} rows in the result DataFrame, but got {all_rows.height} rows."
    )

    # Assert that the sum of feature interaction values per row matches shap_diff within PRECISION_THRESHOLD
    interaction_sums = all_rows.select(list(feature_interaction_dict.keys())).sum_horizontal().to_numpy()
    shap_diffs = np.array(log_odds_predictions) - expected_value
    assert np.all(np.abs(interaction_sums - shap_diffs) < metadata_variables["PRECISION_THRESHOLD"]), (
        "Sum of feature interaction values per row does not match shap_diff within PRECISION_THRESHOLD."
    )

    # Save the DataFrame to feather file
    all_rows.write_ipc(OUTPUT_FILE, compression = "lz4")
    print(f"SUCCESS: Generated data for {cell_line}, UBP IDs {start_id}-{end_id} at {OUTPUT_FILE}.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate SHAP interaction values for a given cell line and set of unique binding pattern IDs.")

    parser.add_argument("--cell_line", type=str, help="Cell line to analyze")
    parser.add_argument("--unique_binding_pattern_ID_range", type=str, help="Range of binding pattern IDs (e.g. '1-101')")
    parser.add_argument("--run_all", action='store_true', help="If set, run for all cell lines and all UBP IDs.")

    args = parser.parse_args()

    if args.run_all:
        
        with open("1_metadata_variables.yaml", 'r') as f:
            metadata_variables = yaml.safe_load(f)
        
        for cell_line in metadata_variables["CELL_LINES"]:
            total_ubp_ids_path = f"{metadata_variables['UBP_ID_TABLE_DIR']}/{cell_line}_unique_binding_pattern_ID_reference_table.tsv.gz"
            total_ubp_id_table = pl.scan_csv(
                total_ubp_ids_path,
                separator="\t", 
            ).select("Unique Binding Pattern ID #").collect()
            
            total_ubp_ids = total_ubp_id_table["Unique Binding Pattern ID #"].to_list()
            assert total_ubp_ids == sorted(total_ubp_ids), f"UBP IDs for {cell_line} are not sorted."

            total_ids_count = len(total_ubp_ids)
            
            batch_size = metadata_variables["SLURM_JOB_BATCH_SIZE"]
            max_ubp_id = max(total_ubp_ids)
            partition_toggle = 0
            
            for start_idx in range(0, total_ids_count, batch_size):
                start_id = total_ubp_ids[start_idx]
                
                if start_idx + batch_size >= total_ids_count:
                    end_id = max_ubp_id  
                else:
                    end_id = total_ubp_ids[start_idx + batch_size - 1]
                
                unique_binding_pattern_ID_range = f"{start_id}-{end_id}"

                OUTPUT_FILE = f"{metadata_variables['SHAP_INTERACTION_VALUES_DIR']}/{cell_line}_UBP_IDs_{start_id}-{end_id}_shap_interactions.feather"

                if not pathlib.Path(OUTPUT_FILE).exists():
                    partition = "standard" if partition_toggle % 2 == 0 else "parallel -N2"
                    partition_toggle += 1

                    os.system(
                        f"sbatch --account=platiglab_paid --partition={partition} -n16 --mem=128GB --time=8:00:00 --out='../SLURM_logs/interaction_value_generation/{cell_line}_UBP_IDs_{start_id}-{end_id}_shap_interactions.out' --error='../SLURM_logs/interaction_value_generation/{cell_line}_UBP_IDs_{start_id}-{end_id}_shap_interactions.err'"
                        f" --wrap='python3.11 {__file__} --cell_line {cell_line} --unique_binding_pattern_ID_range {unique_binding_pattern_ID_range}'"
                    )        

    else: 

        assert args.cell_line is not None, "Please provide --cell_line"
        assert args.unique_binding_pattern_ID_range is not None, "Please provide --unique_binding_pattern_ID_range"

        main(
            args.cell_line, 
            args.unique_binding_pattern_ID_range
        )



