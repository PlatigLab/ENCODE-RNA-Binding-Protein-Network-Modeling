import argparse, requests, pandas as pd, polars as pl, sys, json
from pathlib import Path

from loguru import logger


# Define URL, input file, and output file
io_dir = {

    "Publication-derived PPI w/ ENCODE eCLIP": {
        "url": "https://www.cell.com/cms/10.1016/j.molcel.2024.08.030/attachment/f3ed1198-a29b-4a2d-bb4e-65bda4e8fb61/mmc4.xlsx",
        "Input File": "INPUT_street_et_al_gene_yeo_molecular_cell_table_s3_encode_rbps.xlsx",
        "Output File": "OUTPUT_street_et_al_gene_yeo_molecular_cell_table_s3_encode_rbps.json"
    }, 

    "Full PPI Table": {
        "url": "https://www.cell.com/cms/10.1016/j.molcel.2024.08.030/attachment/5a8579b9-c5d6-40ef-bf06-2753a09cc547/mmc3.csv",
        "Input File": "INPUT_street_et_al_gene_yeo_molecular_cell_table_s2_full_ppi.csv",
        "Output File": "OUTPUT_street_et_al_gene_yeo_molecular_cell_table_s2_full_ppi.json"
    }

}

# Configure logger
logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")


def assert_input_file_exists(io_dir: dict, args: argparse.Namespace):
    """
    Asserts that the input file exists in the specified directory.

    Args:
        io_dir (dict): Dictionary containing input and output file paths.
        args (argparse.Namespace): The command line arguments.
    """

    if args.publication_defined_ENCODE:
        input_file = io_dir["Publication-derived PPI w/ ENCODE eCLIP"]["Input File"]
        url = io_dir["Publication-derived PPI w/ ENCODE eCLIP"]["url"]
    elif args.full_table:
        input_file = io_dir["Full PPI Table"]["Input File"]
        url = io_dir["Full PPI Table"]["url"]
    else:
        raise ValueError("No valid table selection argument provided.")

    # Check if the input file exists
    if not Path(input_file).exists():
        logger.error(f"Input file does not exist.\nPlease download it from {url} and name it as {input_file} in the same directory as the script.")
        sys.exit(1)


def load_PPI_data_from_disk(input_file: str, args: argparse.Namespace) -> pl.DataFrame:
    """
    Loads RBP-RBP PPI information (either full table or just paper-derived ENCODE eCLIP RBPs) into a Polars DataFrame.

    Args:
        input_file (str): Path to the input file.
        args (argparse.Namespace): The command line arguments.

    Returns:
        pl.DataFrame: The loaded Polars DataFrame.
    """    

    if args.publication_defined_ENCODE:

        # Load the Excel file from the URL
        excel_data = pl.read_excel(input_file, sheet_id=0)

        # Process each sheet and add a 'cell_line' column
        dfs = []
        for sheet_name, sheet_data in excel_data.items():
            cell_line = sheet_name.split("_")[2]  # Extract the 3rd element from the sheet name
            sheet_df = sheet_data.with_columns(pl.lit(cell_line).alias("cell_line"))
            dfs.append(sheet_df)

        assert len(dfs) ==2 and dfs[0].shape[0] != dfs[1].shape[0], "The number of rows in the two sheets should be different."
        # Concatenate all sheets into a single DataFrame
        df = pl.concat(dfs)
    
    elif args.full_table:
        raise NotImplementedError("Full table loading is not implemented yet.")

    logger.success(f"Data successfully loaded from URL with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def create_interaction_table_from_publication_defined_ENCODE(df: pl.DataFrame, output_file: str):
    """
    Converts the DataFrame to a JSON file of RBP interactions per cell line.

    Args:
        df (pl.DataFrame): The DataFrame containing the data.
        output_file (str): The path to save the JSON file.
    """

    interaction_dict = {}

    # Iterate over unique cell lines
    for cell_line in df["cell_line"].unique():
        # Subset the DataFrame to the current cell line
        cell_line_df = df.filter(pl.col("cell_line") == cell_line)

        # Create a set of unique interactions
        interactions = set()
        for row in cell_line_df.iter_rows(named=True):
            protein_a, protein_b = row["Protein A"], row["Protein B"]

            # Handle special case for U2AF1 where they made a mistake in the naming
            if ';' in protein_a: 
                protein_a = protein_a.split(';')[1].strip()
                assert protein_a == "U2AF1", f"Unexpected protein name after splitting: {protein_a}"
            if ';' in protein_b:
                protein_b = protein_b.split(';')[1].strip()
                assert protein_b == "U2AF1", f"Unexpected protein name after splitting: {protein_b}"
            
            # Sort the proteins alphabetically and add as a tuple
            interactions.add(
                tuple(
                    sorted(
                        (protein_a.lower(), protein_b.lower())
                        )
                    )
            )

        # Add the interactions to the dictionary under the cell line
        interaction_dict[cell_line] = interactions

    # Sort interactions for each cell line
    for cell_line in interaction_dict:
        interaction_dict[cell_line] = sorted(
            interaction_dict[cell_line],
            key=lambda x: (x[0], x[1])
        )

    # Save the interaction dictionary as a JSON file
    with open(output_file, "w") as f:
        json.dump({k: list(v) for k, v in interaction_dict.items()}, f, indent=4)

    logger.success(f"Interaction table successfully created and saved to {output_file}.")


def main():

    parser = argparse.ArgumentParser(
        description="This script creates JSON files of RBP-RBP PPIs that exist in ENCODE eCLIP data using supplementary table files from Street et al 2024 Molecular Cell (Gene Yeo lab). " \
        "This is achieved either through using publication defined RBPs with ENCODE eCLIP (--publication-defined-ENCODE) or custom processing the full table of RBP-RBP PPIs (--full-table). " \
    )

    parser.add_argument(
        "--publication-defined-ENCODE",
        action="store_true",
        help="For RBP PPIs, use list of RBPs with ENCODE eCLIP as defined in the publication. "
    )
    parser.add_argument(
        "--full-table",
        action="store_true",
        help="Download and process the full RBP-RBP PPI table."
    )

    args = parser.parse_args()
    # Ensure that one of the two boolean arguments is chosen
    if not (args.publication_defined_ENCODE or args.full_table):
        logger.error("You must specify either --publication-defined-ENCODE or --full-table.")
        sys.exit(1)

    # Step 1: Assert that the input file exists
    assert_input_file_exists(io_dir, args)

    if args.publication_defined_ENCODE:
        input_file = io_dir["Publication-derived PPI w/ ENCODE eCLIP"]["Input File"]
        output_file = io_dir["Publication-derived PPI w/ ENCODE eCLIP"]["Output File"]
    elif args.full_table:
        input_file = io_dir["Full PPI Table"]["Input File"]
        output_file = io_dir["Full PPI Table"]["Output File"]

    # Step 2: Load the data from disk
    df = load_PPI_data_from_disk(input_file, args)

    # Step 3: Create the interaction table
    if args.publication_defined_ENCODE:
        create_interaction_table_from_publication_defined_ENCODE(df, output_file)
    elif args.full_table:
        raise NotImplementedError("Full table processing is not implemented yet.")


if __name__ == "__main__":
    main()
    logger.success("Script completed successfully.")