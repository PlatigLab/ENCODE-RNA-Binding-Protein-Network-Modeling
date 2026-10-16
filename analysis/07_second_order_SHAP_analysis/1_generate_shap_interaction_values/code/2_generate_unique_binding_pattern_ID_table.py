import yaml, os
import polars as pl, numpy as np


def main(): 

    with open("./1_metadata_variables.yaml", 'r') as f:
        metadata_variables = yaml.safe_load(f)

    for cell_line in metadata_variables["CELL_LINES"]:

        shap_cache_path = f'{metadata_variables["SHAP_CACHE_DATA"]}/{cell_line}_all-data.feather'
        lf = pl.scan_ipc(shap_cache_path)
        
        columns = lf.collect_schema().names()
        binding_columns = [col for col in columns if col.endswith("_binding")]
        
        unique_df = lf.select(
            binding_columns
        ).unique(
            subset=binding_columns,
            maintain_order=True,
            keep="first"
        ).collect()

        unique_df = unique_df.sort(
            binding_columns, 
            descending=True, 
            maintain_order=True
        )

        original_rows = unique_df.height
        original_columns = unique_df.width

        unique_df = unique_df.with_columns(
            pl.arange(1, unique_df.height + 1).alias("Unique Binding Pattern ID #")
        )
        # Move the ID column to the front
        unique_df = unique_df.select(
            ["Unique Binding Pattern ID #"] + binding_columns
        )

        assert unique_df.height == original_rows, "Number of rows changed after adding Unique Binding Pattern ID"
        assert unique_df.width == original_columns + 1, "Number of columns incorrect after adding Unique Binding Pattern ID"

        assert np.all(
            np.isin(
                unique_df.select(binding_columns).to_numpy(), [0, 1]
            )
        ), "Values other than 0 or 1 found for binding columns"

        assert unique_df.null_count().sum_horizontal().item() == 0, "Null values found in the DataFrame"

        unique_df.write_csv(
            f"{metadata_variables['UBP_ID_TABLE_DIR']}/{cell_line}_unique_binding_pattern_ID_reference_table.tsv", 
            separator="\t"
        )

        os.system(
            f"gzip -f {metadata_variables['UBP_ID_TABLE_DIR']}/{cell_line}_unique_binding_pattern_ID_reference_table.tsv"
        )

    print("SUCCESS: Generated unique binding pattern ID reference tables for all cell lines.")
        

if __name__ == "__main__":

    if "SLURM_JOB_ID" in os.environ:
        main()
    
    else:   
        sbatch_prefix = "sbatch --account=platiglab --partition=parallel -N2 -n16 --mem=128GB --output='../SLURM_logs/generate_unique_binding_pattern_ID_table.out' --error='../SLURM_logs/generate_unique_binding_pattern_ID_table.err'"

        os.system(
            f"{sbatch_prefix} --wrap 'python3.11 {__file__}'",
        )
