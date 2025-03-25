📂 `1_create_RBP_ML_input/`: 

Creates a dataset (1 for each `cell line`) where all splicing events across all samples are individually associated with the number of `eCLIP` peaks of an `RBP` for all `RBPs` and based on a set distance threshold (i.e. peak cannot be too far from splice junction).

Steps are as follows: 
1. Get raw counts expression matrices for each `shRNA + control RNA-Seq` sample (per cell line). 
2. Normalize the raw counts matrices by multiple methods. 
3. Take `eCLIP` peaks and assign them to the closest `RNA splice junction`. 
4. Per cell line, create dataset with number of peaks of an RBP within a set distance threshold of a splice junction. This is done across all samples' splicing events, across all RBPs available, and across all splice junction distances. 


📂 `2_input_binding_exploration/`: 

Before using any machine learning approaches, explore the data to uncover interesting biological patterns. 

📂 `3_choose_dataset_and_model_parameters/`: 

Run through various combinations of dataset parameter and model hyperparameter settings to determine reasonable choices for these parameter types in our project.