import sys, itertools

import polars as pl, pandas as pd

from pathlib import Path
from loguru import logger


ECLIP_RBP_DIR = Path("../01_create_RBP_ML_input/3_assign_eCLIP_to_splice_junctions/output/eclip_and_shrna_rbps/")
PPI_DIR = Path("../../inputs/RBP-RBP_PPI/")
CELL_LINES = ["K562", "HepG2"]


def get_eclip_rbps_for_cell_line(cell_line):
    file_path = ECLIP_RBP_DIR / f"{cell_line}_eclip_rbps.txt"
    with open(file_path, "r") as f:
        rbps = [line.strip().lower() for line in f]
    assert "" not in rbps, "Empty lines found in RBP list"

    assert len(rbps) == len(set(rbps)), "Duplicate RBPs found in RBP list"
    return set(rbps)


def define_eCLIP_RBP_2_order_combinations():
    
    all_combos = []
    for cell_line in CELL_LINES:
        rbps = get_eclip_rbps_for_cell_line(cell_line)

        rbps_sorted = sorted(list(rbps))
        combos = list(itertools.combinations(rbps_sorted, 2))
        
        for combo in combos:
            assert list(combo) == sorted(combo), combo
        
        all_combos.extend(combos)
    
    unique_combos = list(set(all_combos))
    return unique_combos


def build_rec_y2h_table(combos): 

    logger.warning(
        (
            "REMINDER: when buiding Rec-Y2H table: since paper does not provide all "
            "possible interaction results and they screen ~98-99% of possible RBP-RBP pairs, "
            "we are assuming that any pair not listed as a positive hit is a negative hit."
        )
    )
    
    screened_rbps = pl.read_excel(
        PPI_DIR / "Rec-Y2H" / "all_rbps_screened.xlsx", 
    )["Gene Symbol"].str.to_lowercase().to_list()
    unique_screened_rbps = set(screened_rbps)

    assert "" not in unique_screened_rbps, "Empty RBP names found in screened RBPs"

    # SumIS >= 7.1 is the threshold for positive interactions as defined in the Rec-Y2H paper.
    results = pl.read_excel(
        PPI_DIR / "Rec-Y2H" / "lang_et_al_rec-y2h_screening_results.xlsx", 
    ).filter(pl.col("sumIS") >= 7.1)

    # Build positive hits dictionary
    positive_hits = {}
    protein_a = results["Protein A"].str.to_lowercase().to_list()
    protein_b = results["Protein B"].str.to_lowercase().to_list()

    for a, b in zip(protein_a, protein_b):
        if a not in positive_hits:
            positive_hits[a] = set()
        if b not in positive_hits:
            positive_hits[b] = set()

        positive_hits[a].add(b)
        positive_hits[b].add(a)

    all_data = []
    for rbp1, rbp2 in combos:
        assert rbp1 < rbp2, f"{rbp1} should precede {rbp2} in string sorting order"

        interaction = f"{rbp1}-{rbp2}"

        # Check if both RBPs were screened
        if rbp1 not in unique_screened_rbps or rbp2 not in unique_screened_rbps:
            rec_y2h = None
        else:
            # Check for positive interaction
            found = (rbp1 in positive_hits and rbp2 in positive_hits[rbp1])
            
            if found: 
                assert (rbp2 in positive_hits and rbp1 in positive_hits[rbp2]), \
                    f"Inconsistent positive interaction: {rbp1}-{rbp2} found but not {rbp2}-{rbp1}"

            rec_y2h = True if found else False

        row = {
            "Interaction": interaction,
            "RBP 1": rbp1,
            "RBP 2": rbp2,
            "Rec-Y2H": rec_y2h,
        }

        all_data.append(row)

    df = pl.DataFrame(all_data)
    return df


def build_street_et_al_table(combos): 
    MODES = {
        'Street et al | IP-MS': "IP", 
        "Street et al | IP/SEC-MS (Both)": "both"
    }

    logger.warning("REMINDER: Street et al. table construction uses synonyms for 2 RBPs that we have data for, which are U2AF1 and EIF3H. Manually setting those synonyms back to eCLIP RBP names.")

    # need to skip first row and then second row becomes header
    all_rbps_screened = pd.read_excel(
        PPI_DIR / "Street_et_al" / "INPUT_street_et_al_gene_yeo_molecular_cell_table_s1_ip_ms_baits.xlsx",
        sheet_name="All baits",
        header=1
    )["Bait"].str.lower().tolist()
    unique_all_rbps_screened = set(all_rbps_screened)  

    assert "" not in unique_all_rbps_screened, "Empty RBP names found in screened RBPs"
    assert len(all_rbps_screened) == len(unique_all_rbps_screened), "Duplicate RBPs found in screened RBPs"

    results_lf = pl.scan_csv(
        PPI_DIR / "Street_et_al" / "INPUT_street_et_al_gene_yeo_molecular_cell_table_s2_full_ppi.csv"
    )

    # Iterate over interaction types and build dictionaries of partners
    interaction_dicts = {}
    for mode, interaction_type in MODES.items():
        df = (
            results_lf
            .filter(pl.col("Interaction_support") == interaction_type)
            .with_columns([
                pl.col("Bait").str.to_lowercase().alias("Bait"),
                pl.col("Prey").str.to_lowercase().alias("Prey"),
            ])
            .select(["Bait", "Prey"])
            .collect()
        )

        partners_dict = {}
        # Clean up bait and prey columns 
        bait_list = [
            "u2af1" if "u2af1" in bait else "eif3h" if "eif3h" in bait else bait
            for bait in df["Bait"].to_list()
        ]
        prey_list = [
            "u2af1" if "u2af1" in prey else "eif3h" if "eif3h" in prey else prey
            for prey in df["Prey"].to_list()
        ]
        
        # synonyms/aliases for the proteins that we have in our dataset should not be there anymore 
        assert not any("u2af1l5" in s for s in bait_list + prey_list), "Found string containing 'u2af1l5'"
        assert not any("eif3s3" in s for s in bait_list + prey_list), "Found string containing 'eif3s3'"
        
        for bait, prey in zip(bait_list, prey_list):
            if bait not in partners_dict:
                partners_dict[bait] = set()
            if prey not in partners_dict:
                partners_dict[prey] = set()
            
            assert prey != bait, f"Found self-interaction for {bait}"

            partners_dict[bait].add(prey)
            partners_dict[prey].add(bait)
        
        interaction_dicts[mode] = partners_dict

    all_data = []
    for rbp1, rbp2 in combos:
        assert rbp1 < rbp2, f"{rbp1} should precede {rbp2} in string sorting order"
        interaction = f"{rbp1}-{rbp2}"

        # Check if either RBP was a screened bait
        if rbp1 not in unique_all_rbps_screened and rbp2 not in unique_all_rbps_screened:
            ip_ms = None
            both_mode = None

        else:
            ip_dict = interaction_dicts['Street et al | IP-MS']
            both_dict = interaction_dicts['Street et al | IP/SEC-MS (Both)']

            # For IP-MS mode: check in both IP-MS and Both categories (union as mutually exclusive)
            found_ip = (rbp1 in ip_dict and rbp2 in ip_dict[rbp1])
            found_both = (rbp1 in both_dict and rbp2 in both_dict[rbp1])
            ip_ms = True if (found_ip or found_both) else False

            # For Both mode: only check in Both category 
            both_mode = True if found_both else False

            if both_mode: 
                assert rbp2 in both_dict and rbp1 in both_dict[rbp2], \
                    f"Inconsistent IP-MS interaction: {rbp1}-{rbp2} found but not {rbp2}-{rbp1}"
            elif ip_ms: 
                assert rbp2 in ip_dict and rbp1 in ip_dict[rbp2], \
                    f"Inconsistent IP-MS interaction: {rbp1}-{rbp2} found but not {rbp2}-{rbp1}"

        row = {
            "Interaction": interaction,
            "RBP 1": rbp1,
            "RBP 2": rbp2,
            "Street et al | IP-MS": ip_ms,
            "Street et al | IP/SEC-MS (Both)": both_mode,
        }

        all_data.append(row)

    df = pl.DataFrame(all_data)
    assert not ((df["Street et al | IP-MS"] == False) & (df["Street et al | IP/SEC-MS (Both)"] == True)).any(), \
        "Found rows where 'Street et al | IP-MS' is False but 'Street et al | IP/SEC-MS (Both)' is True"
    
    return df


def join_tables_by_interaction(table1, table2, combos):
    
    # Outer join on "Interaction", "RBP 1", "RBP 2"
    joined = table1.join(table2, on=["Interaction", "RBP 1", "RBP 2"], how="full", validate="1:1", coalesce=True)

    # Identify columns unique to each table
    unique_table1_cols = set(table1.columns) - set(table2.columns)
    unique_table2_cols = set(table2.columns) - set(table1.columns)

    # Count nulls before join
    nulls_table1 = {col: table1[col].null_count() for col in unique_table1_cols}
    nulls_table2 = {col: table2[col].null_count() for col in unique_table2_cols}

    # Count nulls after join
    nulls_joined_table1 = {col: joined[col].null_count() for col in unique_table1_cols}
    nulls_joined_table2 = {col: joined[col].null_count() for col in unique_table2_cols}

    # Check that null counts are unchanged
    for col in unique_table1_cols:
        assert nulls_table1[col] == nulls_joined_table1[col], f"Null count changed for {col} from table1"
    for col in unique_table2_cols:
        assert nulls_table2[col] == nulls_joined_table2[col], f"Null count changed for {col} from table2"

    # Check row count
    assert joined.height == len(combos), f"Joined table row count {joined.height} != combos length {len(combos)}"
    logger.info(f"Joined table has {joined.height} rows corresponding to {len(combos)} unique RBP-RBP combinations.")

    return joined


def add_eCLIP_availability_annotation(table):
    # Get eCLIP RBPs per cell line
    eclip_rbps_per_cell_line = {cell_line: get_eclip_rbps_for_cell_line(cell_line) for cell_line in CELL_LINES}

    # Add columns for each cell line indicating if both RBPs are in eCLIP set
    for cell_line in CELL_LINES:
        table = table.with_columns(
            (
                (pl.col("RBP 1").is_in(eclip_rbps_per_cell_line[cell_line])) &
                (pl.col("RBP 2").is_in(eclip_rbps_per_cell_line[cell_line]))
            ).alias(f"{cell_line} - Both eCLIP")
        )
        
        assert table[f"{cell_line} - Both eCLIP"].null_count() == 0, f"Nulls found in '{cell_line} - Both eCLIP' column"

    # Assert that there are no rows where both "K562 - Both eCLIP" and "HepG2 - Both eCLIP" are False
    assert not ((~table["K562 - Both eCLIP"]) & (~table["HepG2 - Both eCLIP"])).any(), \
        "Found rows where both 'K562 - Both eCLIP' and 'HepG2 - Both eCLIP' are False"

    return table


def output_final_table(table): 

    # Check for swapped duplicates (permutations) in the table
    duplicates = table.filter(
        (pl.col("RBP 1") >= pl.col("RBP 2"))
    )
    assert duplicates.height == 0, "Found rows where RBP 1 > RBP 2 (permuted pairs present, expected only combinations)"

    # Uppercase the relevant columns before output
    table = table.with_columns([
        pl.col("Interaction").str.to_uppercase().alias("Interaction"),
        pl.col("RBP 1").str.to_uppercase().alias("RBP 1"),
        pl.col("RBP 2").str.to_uppercase().alias("RBP 2"),
    ]).sort(["RBP 1", "RBP 2"])

    table.write_csv(
        "./all_eCLIP_RBP_PPI_combination_PPI_annotations.tsv",
        separator="\t",
    )


if __name__ == "__main__":

    combos = define_eCLIP_RBP_2_order_combinations()
    rec_y2h_table = build_rec_y2h_table(combos)
    street_et_al_table = build_street_et_al_table(combos)
    joined_table = join_tables_by_interaction(rec_y2h_table, street_et_al_table, combos)
    final_table = add_eCLIP_availability_annotation(joined_table)
    output_final_table(final_table)


    logger.success("Script completed successfully.")