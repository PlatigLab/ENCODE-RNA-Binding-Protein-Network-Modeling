## Analyses

### 05_08_ctrl_shap_all_data.cutoffs.csv
Table S1: Compares dPSI and CTRL SHAP across all features, in both cell lines, using 4 different SHAP cutoffs

### 05_08_ctrl_shap_each_feature.csv
Compares dPSI and CTRL SHAP for each individual feature in both cell lines

### final_results_table.ipynb
Concatnates the individual results files from getting SHAP values and dPSIs for the CRISPR KD RBPs

### supp_table_S1.ipynb
Creates the all_data and each_feature csv files

## Figures
### figure_2_B_generation.ipynb
Code to generate figure 2B

## Results
### HepG2_individual_results
Each Crispr RBPs' individual results after retreiving SHAP and dPSI

### K562_individual_results
Each Crispr RBPs' individual results after retreiving SHAP and dPSI

## HepG2 & K562
All code to do the Crispr RBP analysis and the precomputed rMATS files


### how to use pipeline

1. Pre-process the BATs

- Use the preprocess_.sh to run the .py file that preprocesses the BAT
- Edit the script to include all RBPs of interest (those with Crispr KD, eCLIP, NO sHRNA)

2. Use the CL_Slurm script to actually run the pipeline

- It calls the Crispr_KD script and actually performs the analysis