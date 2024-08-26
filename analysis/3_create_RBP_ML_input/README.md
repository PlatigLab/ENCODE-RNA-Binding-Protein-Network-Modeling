📂 `1_get_expression_from_BAMs/`: 

Code to get raw expression for each `shRNA BAM` file and then convert them to a cleaned up raw counts matrix for each cell line. 

📂 `2_normalize_raw_counts_matrices/`: 

Convert raw counts matrices into normalized counts matrices using multiple different methods.

(Obviously, expression matrices are separated by cell line). 

📂 `3_assign_eCLIP_to_splice_junctions/`: 

Get `BED` file of unique `Skipped Exon splice junctions` and `BED` file of all `eCLIP` data (per cell line) to assign `bedtools closest` splice junction for each `eCLIP` peak. 

📂 `4_create_num_peaks_ML_input`:

Create ML input data for the "number of peaks" flavor of the data (per cell line and distance threshold) where **importantly**, this data is used as a precursor to generate all other flavors of the data.