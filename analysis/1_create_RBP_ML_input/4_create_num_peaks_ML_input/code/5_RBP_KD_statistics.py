import polars as pl
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

DATA_PATH = "../../../../../../../data/RBP_ML/3_yogi_dataset_feb_2025/"
OUTPUT_PATH = "../corrected_has_RBP_KD_output/"
CELL_LINES = ["K562", "HepG2",]


def process_row(row):
    new_row = {"index": row["index"]}
    rbp_kd_target = row["RBP_KD_Target"]

    if rbp_kd_target != "CTRL":
        # Add new columns for each position and the overall flag
        for pos in range(1, 7):
            binding_col = f"{rbp_kd_target}_{pos}_binding"

            if row[binding_col] >= 1:
                new_row[f"has_RBP_KD_{pos}"] = True
            elif row[binding_col] == 0:
                new_row[f"has_RBP_KD_{pos}"] = False
            else:
                assert False, f"Unexpected value in {binding_col}: {row[binding_col]}"
        # Add the 7th column to indicate if any of the 6 positions have a KD
        new_row[f"has_RBP_KD"] = any(new_row[f"has_RBP_KD_{pos}"] for pos in range(1, 7))

    elif rbp_kd_target == "CTRL":
        # Set all new columns to False for "CTRL"
        for pos in range(1, 7):
            new_row[f"has_RBP_KD_{pos}"] = False
        new_row[f"has_RBP_KD"] = False
    else: 
        assert False, f"Unexpected RBP_KD_Target value: {rbp_kd_target}"

    return new_row


def main(cell_line):
    input_file = f"{DATA_PATH}/{cell_line}_100_all-events_num-peaks-no-kd.tsv.gz"

    # Load the input file
    df = pl.scan_csv(input_file, separator="\t")
    names = df.collect_schema().names()
    
    # Select relevant columns
    columns_to_select = ["index", "RBP_KD_Target"] + [col for col in names if col.endswith("_binding")]
    df = df.select(columns_to_select).collect()

    # Use ThreadPoolExecutor to parallelize row processing
    with ThreadPoolExecutor() as executor:
        processed_rows = list(tqdm(executor.map(process_row, df.iter_rows(named=True)), desc="Processing rows", total=df.height))
    
    # Create a new DataFrame from the processed rows
    df = pl.DataFrame(processed_rows)
    # Assert that there are no missing or null values
    assert df.null_count().sum_horizontal().item() == 0, "There are missing or null values in the DataFrame."

    # Save the output DataFrame
    output_file = f"{OUTPUT_PATH}/{cell_line}_corrected_has_RBP_KD.feather"
    df.write_ipc(output_file)


if __name__ == "__main__":

    for cell_line in CELL_LINES:
        main(cell_line)
