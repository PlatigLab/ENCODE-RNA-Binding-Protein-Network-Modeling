📂 `1_batch_effect_investigation/`: 

Comparing how much batch effects can be found in PCA data using `PSI` aka `IncLevel` as matrix values. 

Sub-folders separated by notebooks and output. 

Output folders are mixture of `genome build` and values used in the matrix (i.e. `counts` vs `PSI`)

📂 `2_Ayan_input_binding_data_exploration/`: 

Taking input binding matrices that @talismanbrandi has created and exploring combinatorial binding patterns of `RBPs` to `Skipped Exon` events. 

This is both for the July 2024 Gordon Conference on Post-Transcriptional Gene Regulation as well as creating a baseline for comparison against our model in our upcoming publication. 

📂 `3_create_RBP_ML_input/`: 

Series of individual steps that ultimately creates our own version of the RBP ML input binding data. 

Steps are as follows: 
1. Get raw counts expression matrices for each `shRNA + control RNA-Seq` sample (per cell line). 
2. Normalize the raw counts matrices by multiple methods. 
3. Take `eCLIP` peaks and assign them to the closest `RNA splice junction`. 
4. Create a precursor for ML input data by outputting, per cell line and distance threshold, the number of peaks of an RBP within a set distance threshold of a splice junction. 