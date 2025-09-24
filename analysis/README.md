📂 `01_create_RBP_ML_input/`: 

Creates a dataset (1 for each `cell line`) where all splicing events across all samples are individually associated with the number of `eCLIP` peaks of an `RBP` for all `RBPs` and based on a set distance threshold (i.e. peak cannot be too far from splice junction).

Steps are as follows: 
1. Get raw counts expression matrices for each `shRNA + control RNA-Seq` sample (per cell line). 
2. Normalize the raw counts matrices by multiple methods. 
3. Take `eCLIP` peaks and assign them to the closest `RNA splice junction`. 
4. Per cell line, create dataset with number of peaks of an RBP within a set distance threshold of a splice junction. This is done across all samples' splicing events, across all RBPs available, and across all splice junction distances. 


📂 `02_input_binding_exploration/`: 

* Checks `PSI` distributions across cell lines. 
* Explores RBP binding preferences and percentages. 
* Displays training strategy tables
* HACKY ANALYSES for Pete's UNC Talk in 2025: 
  * ElasticNet, XGBoost, performance, SHAP. 

📂 `03_choose_dataset_and_model_parameters/`: 

* Establish dataset hyperparameters
* Run sequential, interrelated hyperparameter sweeps for XGBoost models. 
* Run ElasticNet models. 
* Plot performance metrics in relation to other important covariates to understand what influences model performance.

📂 `04_run_final_models_and_SHAP/`: 
  
* Replicate [PlatigLib](https://github.com/PlatigLab/platiglib) model in my own way.
* Run `SHAP` analysis with trained model.
* Run experiments for how changing background data changes `SHAP` results.

📂 `05_retrieve_gene_yeo_RBP_PPIs_street_et_al_molecular_cell/`: 

Contains input data and code to retrieve RBP-RBP protein interaction data from [Street et al. 2024 Molecular Cell](https://doi.org/10.1016/j.molcel.2024.08.030) and save as a `JSON` object for downstream analysis and as a lab resource.

📂 `analysis/06_SHAP_network_analysis/`: 

Too many analyses to list here but in summary, this contains the bulk of the `SHAP` analyses that went into the paper and most of the important figures in the paper. 

**NOTE**: Non-`SHAP` analyses (`ElasticNet` and `Model Performance`) are also included here.

📂 `07_reece_biological_validations/`: 

Contains Reece's analyses:
* Volcano plots to compare global shap at a position and dPSI
* Correlations of local SHAP values in PPI and non-PPI pairs