📂 `1_batch_effect_investigation/`: 

Comparing how much batch effects can be found in PCA data using `PSI` aka `IncLevel` as matrix values. 

Sub-folders separated by notebooks and output. 

Output folders are mixture of `genome build` and values used in the matrix (i.e. `counts` vs `PSI`)

📂 `2_input_binding_data_exploration/`: 

Taking input binding matrices that @talismanbrandi has created and exploring combinatorial binding patterns of `RBPs` to `Skipped Exon` events. 

This is both for the July 2024 Gordon Conference on Post-Transcriptional Gene Regulation as well as creating a baseline for comparison against our model in our upcoming publication. 

📂 `3_get_expression_shrna_BAMs/`: 

Code to get raw expression for each `shRNA BAM` file and then convert them to a cleaned up raw counts matrix for each cell line. 

📂 `4_normalize_raw_counts_matrices/`: 

Convert raw counts matrices into normalized counts matrices using multiple different methods.

(Obviously, expression matrices are separated by cell line). 

📂 `5_assign_eCLIP_to_splice_junctions/`: 

Get `BED` file of unique `Skipped Exon splice junctions` and `BED` file of all `eCLIP` data (per cell line) to assign `bedtools closest` splice junction for each `eCLIP` peak. 