
<h1 style="color: teal; font-weight: bold; font-size: 30px;">All commands here assume you are in the root directory of this project.</h1>
<h1 style="color: teal; font-weight: bold; font-size: 30px;">Figure and Table reproduction is found {{{{{{{{{{{TODO}}}}}}}}}}}.</h1>


‼️**NOTE: analyses were done using `Python 3.11.11`.**‼️


# 1 - Data Creation
‼️**NOTE: we only consider the autosomal chromosomes (chr1-chr22) for ALL analyses in this project.**‼️

## 1.1 -  Assign `eCLIP` peaks to splice junctions 


‼️**Before running the script below:**‼️

* Set the `SE_GLOB` variable to where **all Skipped Exon *(SE)* files** are within **each RBP's subfolder**
    * The expected structure is: 
        *  `/path/to/rMATS/results/{RBP}-{Batch}-{Cell Line}/SE.MATS.JC.txt`

* Set the `PEAKS_PATH` variable to where the `ENCODE eCLIP peaks` are located
    * The expected structure is the same as downloading from the `ENCODE` portal whereby all files are listed flat in one folder. 

* Download and install [`bedtools`](https://github.com/arq5x/bedtools2/releases) if you do not have it already as it is required for the script below (version `2.31.1` was used in this analysis).
    * Update the `BEDTOOLS_PATH` variable in the script below to the path of the `bedtools` executable on your machine.

```
cd ./analysis/01_create_RBP_ML_input/3_assign_eCLIP_to_splice_junctions/code/

python3.11 eclip_assigner.py
```

## 1.2 - Create `splice instances` per each unique combination of `Cell Line` & `RBP` at various distance thresholds

* Before running below script, set the `RMATS_DATA_PATH` to the same path as `SE_GLOB` variable in the previous script. 

* We use the `100` distance threshold for this entire project.
    * This means `100` on either side of the splice junction (i.e. `200` total) is the window within which we consider `RBP binding` to be potentially influential for splicing outcomes.

```
cd ./analysis/01_create_RBP_ML_input/4_create_num_peaks_ML_input/code/

bash 2_submission_script.sh
```

## 1.3 -  Concatenate `splice instances` across all RBPs per each `cell line` and distance threshold

#### NOTES: 

1. This script removes duplicate `splice instances` (rows) in dataset that are caused by specific `control shRNA RNA-Seq` samples being used as controls for multiple different `RBP KD RNA-Seq` samples.
    * This is intentionally a part of the `ENCORE` consortium's experimental design. 
    * ‼️**A tricky nuance that happens here is that after de-duplication the columns `Raw P-val`, `FDR`, `DeltaPSI` values are no longer meaningful since these values were originally calculated based on the specific `RBP KD` it was compared against.**
        * Hence, for the 3 columns that you can't trust, only use rows that came from `RBP KD RNA-Seq` samples and if you need to figure out which `RBP KD` and which `rMATS row number` that a given control row's 3 columns' values came from, you can use the `Associated Experiment` and `rMATS Event ID` columns.
2. ‼️The dataset created here, and as labelled with the suffix of the files as `num-peaks-no-kd.tsv`, contains the number of peaks for a given RBP at a given position and has not been subjected to any `in-silico KD` procedure. ‼️
    * Data QC and final dataset creation steps are described [later](#2---pre-modeling-data-exploration).

```
cd ./analysis/01_create_RBP_ML_input/4_create_num_peaks_ML_input/code/

sbatch .3_concatenate_results.sh
```

## 1.4 -  Run data QC assertions and assign `exon IDs` to each `splice instance`

* Before running below script, set the following variables: 
    * `OUTPUT_DIR` & `DATA_PATH`
        * These should be the same path and be located outside this repository
    * `DATA_VERSION`
```
cd ./analysis/01_create_RBP_ML_input/4_create_num_peaks_ML_input/code/

python3.11 4_exon_assignment_and_data_assertions.py
```

## 1.5 - Get `RBP in-silico KD` information for each `splice instance` at each `position`
#### NOTE: For each `splice instance` (row) from an `RBP KD RNA-Seq` sample, we see if the `KD RBP` is originally bound in the `eCLIP` data at any of the `6 positions` and, in a SEPARATE table, we set `True` if so and `False` otherwise. Rows from `control shRNA RNA-Seq` samples are obviously always set to `False`. 

Before running below script, set `DATA_PATH` to the same path as the `DATA_PATH` variable from the script in the previous step.

```
cd ./analysis/01_create_RBP_ML_input/4_create_num_peaks_ML_input/code/

python3.11 5_RBP_KD_statistics.py
```

<br>
<br>

# 2 - Pre-Modeling Data Exploration
## 2.1 - Across both `cell lines`, explore `PSI` distributions and `RBP Binding` preference/frequency

* Set `DATA_PATH` variable in the following file to be the same path as the `DATA_PATH` variable from the script in the previous step: 
    * `./analysis/02_input_binding_exploration/code/1_binding_analyzer.py`

‼️**IMPORTANT: the notebook below automatically chooses the "distance to splice junction" threshold as 100 base pairs and only keeps rows with 40 or more sequencing reads (i.e. the PSI calculation came from having at least 40 reads). This MUST be run because the `read_data()` function creates our final dataset that is used for all analyses downstream.**‼️

```
Run notebook from top to bottom: 

cd ./analysis/02_input_binding_exploration/code/

jupyter notebook 1_yogi_binding_analyzer.ipynb
```

<br>
<br>

# 3 - `XGBRegressor` and `ElasticNet` Hyperparameter Tuning & Subsequent Results Exploration

## 3.1- Set up `platiglib`
```
git clone https://github.com/PlatigLab/platiglib

# specific commit that was used in this analysis
git checkout 1ba3ba134aa8acba65d1a97e39a2800ff3cacb16
```

Set up virtual environment and variables as described here: https://github.com/PlatigLab/platiglib/blob/1ba3ba134aa8acba65d1a97e39a2800ff3cacb16/docs/quickstart.md#installation

‼️**IMPORTANT: if you do not follow the below steps, you will not be able to run this sections' analyses**‼️

* Analyses in this section that require `platiglib` MUST be run within the virtual environment setup (details in the previous link). 
* Set the scripts directory (the `scripts/` folder within `platiglib`) variable as an absolute path across the many `Python` files mentioned in this section.
* Make sure the sweep configuration `YAML` files passed in have `name` and `project` under the `sweep` top-key properly set. 👇
    ```
        sweep:
        name: <SWEEP_NAME>
        project: <WANDB_PROJECT_NAME>

        {rest of YAML file...}
    ```

## 3.2 - Create `platiglib_cache` data cache before model training

```
cd ./analysis/03_choose_dataset_and_model_parameters/code/

python3.11 1_submit_data_caching_sweep.py
```

## 3.3 - Hyperparameter Tuning

### 3.3.1 - Sequential, interrelated hyperparameter tuning strategy (`XGBRegressor`)

#### NOTE: the idea here is to make hyperparameter tuning more feasible by grouping hyperparameters together that most influence each other and organizing these based on which groups of hyperparameters are most to least important (e.g. we tune a set of hyperparameters that are more important than the next set of hyperparameters and then fix the first set of hyperparameters to the best values before tuning the next set of lesser-important hyperparameters).

#### 3.3.1.1 - 1st set: `learning_rate`, `max_depth`, `early_stopping_rounds`
NOTE: it is not common to sweep `early_stopping_rounds` but for our own exploration we did so. 

**ALL subsequent sweeps used `early_stopping_rounds=100` (a canonical value)**

```
cd ./analysis/03_choose_dataset_and_model_parameters/code/

python3.11 2_tree_construction_parameters.py
```

#### 3.3.1.2 - 2nd set: `min_child_weight`, `gamma`

**NOTE: From previous sweep, take best performing swept hyperparameters by average `holdout_r2_score` across the seeds and per `cell line` to include here:**


* `HepG2`: 
    * `./analysis/03_choose_dataset_and_model_parameters/wandb_configs/model/2_sequential_sweeps/2_tree_splitting_parameters_HepG2.yaml`
* `K562`: 
    * `./analysis/03_choose_dataset_and_model_parameters/wandb_configs/model/2_sequential_sweeps/2_tree_splitting_parameters_K562.yaml`

After, you can run: 

```
cd ./analysis/03_choose_dataset_and_model_parameters/code/

python3.11 3_tree_splitting_parameters.py
```        

#### 3.3.1.3 - 3rd set: `colsample_bytree`, `subsample`

**NOTE: From previous sweep, take best performing swept hyperparameters by average `holdout_r2_score` across the seeds and per `cell line` to include here:**

* `HepG2`: 
    * `./analysis/03_choose_dataset_and_model_parameters/wandb_configs/model/2_sequential_sweeps/3_data_sampling_HepG2.yaml`
* `K562`: 
    * `./analysis/03_choose_dataset_and_model_parameters/wandb_configs/model/2_sequential_sweeps/3_data_sampling_K562.yaml`

After, you can run:

```
cd ./analysis/03_choose_dataset_and_model_parameters/code/

python3.11 4_data_sampling.py
```

#### 3.3.1.4 - 4th set: `reg_alpha`, `reg_lambda`

**NOTE: From previous sweep, take best performing swept hyperparameters by average `holdout_r2_score` across the seeds and per `cell line` to include here:**

* `HepG2`: 
    * `./analysis/03_choose_dataset_and_model_parameters/wandb_configs/model/2_sequential_sweeps/4_regularization_HepG2.yaml`
* `K562`:
    * `./analysis/03_choose_dataset_and_model_parameters/wandb_configs/model/2_sequential_sweeps/4_regularization_K562.yaml`

After, you can run:
```
cd ./analysis/03_choose_dataset_and_model_parameters/code/

python3.11 5_regularization.py
```

### 3.3.2 - `ElasticNet` hyperparameter tuning

#### NOTE: We use a different `W&B` project for linear models as the `W&B` results summary table has a different schema than the `W&B` summary table for `XGBRegressor` models. 

```
cd ./analysis/03_choose_dataset_and_model_parameters/code/

python3.11 7_run_elasticnet.py
```

## 3.4 - Hyperparameter Tuning Results Exploration

#### NOTE: make sure to set the `W&B Project` names within the `sweep_projects` variable of the analysis class file here: `./analysis/03_choose_dataset_and_model_parameters/code/dataset_and_model_parameter_chooser.py`

You can then run the following notebook and all cells except for those that contain the functions [1] `assert_outer_loop_holdout_r2_scores()`, [2] `inner_fold_vs_outer_fold_r2_score()`, [3] `plot_elasticnet_outer_vs_avg_inner_r2()`, [4] `create_model_parameter_configs_for_replication()`: 

```
./analysis/03_choose_dataset_and_model_parameters/code/choosing_parameters.ipynb
```

‼️**To continue to the next section, you must make sure to execute code cells that contain the following functions**‼️

* `get_run_ids_for_outer_loop_holdout_r2_scores()`
* `get_elasticnet_run_ids()`

## 3.5 - Retrieve model performance on data never used during tuning (`final holdout "test" dataset`)

#### NOTE: make sure to update the `PROJECTS` dictionary variable in the below script and specifically set the `project` sub-keys to the correct `W&B` project names. 

After, you can run the following: 

```
cd ./analysis/03_choose_dataset_and_model_parameters/code/

python3.11 6_run_outer_loop_holdout_r2.py
```

To view the results from the above script, go back to the previous notebook (`./analysis/03_choose_dataset_and_model_parameters/code/choosing_parameters.ipynb`) and run the cells containing the following functions: 

* `assert_outer_loop_holdout_r2_scores()`
* `inner_fold_vs_outer_fold_r2_score()`
* `plot_elasticnet_outer_vs_avg_inner_r2()`

## 3.6 - Output full configuration of parameters and hyperparameters needed to produce the final set of top models 

Within the aforementioned notebook (`./analysis/03_choose_dataset_and_model_parameters/code/choosing_parameters.ipynb`), run the cell containing the following function to output the final set of parameters and hyperparameters for the top models per cell line and model type:

* `create_model_parameter_configs_for_replication()`

<br>
<br>

# 4 - Final Model Training & `Individual Feature Effects SHAP` Value Generation
## 4.1 - Train top 5 models for `XGBRegressor` and `ElasticNet` for both `HepG2` and `K562` (20 models total)

### 4.1.1 - Create data splits for final model training and evaluation

The following script creates the `train`, `validation`, and `test` splits for the final model training and is meant to make sure that the same data splits are used across the different models trained in the next step. 

```
cd ./analysis/04_run_final_models_and_SHAP/code/

python3.11 1_replicate_platiglib_model.py --split_data
```

### 4.1.2 - Train final models for each `cell line` and model type

```
cd ./analysis/04_run_final_models_and_SHAP/code/

python3.11 1_replicate_platiglib_model.py --run_all
```

### 4.1.3 - Generate `Individual Feature SHAP` values for each of the final models trained in the previous step

#### NOTES: 
* `Individual Feature` refers to the fact that we are only generating `SHAP` values for the original features (each `RBP` binding at each `position`) and not generating `SHAP` values for the feature-feature interaction terms (i.e. `Feature Interaction Effect SHAP` values).
    * `Interaction SHAP` is covered in [another section](#7---feature-interaction-effect-shap-analysis).
* `SHAP` values are generated for the `XGBRegressor` models using `tree_path_dependent` algorithm with the units being in `log-odds` space.
* `SHAP` values are generated for all the `Unique Binding Patterns`.

```
cd ./analysis/04_run_final_models_and_SHAP/code/

python3.11 2_run_SHAP_analysis.py --run_all_normal

```

### 4.1.4 - Join `SHAP` values for `Unique Binding Patterns` with original data

**This step takes the `SHAP` values for the `Unique Binding Patterns` generated in the previous step and joins them with the original dataset that contains all the original rows (`splice instances`).**

```
cd ./analysis/04_run_final_models_and_SHAP/code/

python3.11 3_join_unique_binding_SHAP_results_with_all_data.py --run_all normal
```

<br>
<br>

# 5 - Assign `PPI` Status for each RBP-RBP Pair of the `eCLIP` RBPs in either `Cell Line`.

This resource table is needed for downstream analyses that require information about "known" `RBP PPIs`. 

**NOTE**: 
* For `Rec-Y2H` data, we assume that all pairs of bait and preys are sampled because: 
    1. Their original paper states that they sampled between ~98-99% of all possible pairs. 
    2. There's no way to know which pairs were not sampled based on their Supplementary Tables. 

 

```
cd ./analysis/05_generate_eCLIP_RBP_PPIs/

python3.11 retrieve_ppis.py
```

<br>
<br>

# 6 - `Individual Feature SHAP` Analysis

‼️**Make sure to update the `SBATCH_PREFIX` variable in this script:**‼️ 
* `./analysis/06_first_order_SHAP_analysis/code/1_first_order_shap_investigator.py` 

## 6.1 - Run data QC and data cache creation to prepare for downstream analyses

### 6.1.1 - Create data caches

👇 invokes `load_final_SHAP_data(underlying_data="All-Data")`
```
cd ./analysis/06_first_order_SHAP_analysis/code/

python3.11 1_first_order_shap_investigator.py --parallelize create_final_shap_cache
```
Creates tables with: 
* Binding data
* `rMATS` metadata 
* `local SHAP` values
    * **These values are averaged across the top 5 models for that cell line.**
<br>

👇 invokes `plot_dpsi_vs_local_SHAP_for_test_data()`
```
cd ./analysis/06_first_order_SHAP_analysis/code/

python3.11 1_first_order_shap_investigator.py --parallelize create_test_data_dpsi_vs_local_SHAP_scatterplot_data
```

Creates table comparing `deltaPSI` values to `local SHAP` values for the `final holdout "test" dataset`. 

👇 invokes `is_position_3_4_activating_and_others_repressing()`

```
cd ./analysis/06_first_order_SHAP_analysis/code/

python3.11 1_first_order_shap_investigator.py --parallelize is_position_3_4_activating_and_others_repressing
```

Creates long table of each of the local SHAP values from `Bound` features and some associated metadata.

### 6.1.2 - Run data QC assertions 

👇 invokes `run_data_quality_assertions()`
```
cd ./analysis/06_first_order_SHAP_analysis/code/

python3.11 1_first_order_shap_investigator.py --parallelize data_qc_assertions
```

👇 invokes `check_no_SHAP_variance_per_binding_pattern()`
```
cd ./analysis/06_first_order_SHAP_analysis/code/

python3.11 1_first_order_shap_investigator.py --parallelize no_shap_variance_per_binding_pattern
```

👇 invokes `assert_SHAP_additivity()`
```
cd ./analysis/06_first_order_SHAP_analysis/code/

python3.11 1_first_order_shap_investigator.py --parallelize shap_additivity_assertions
```

## 6.2 - Calculate per-feature `SHAP` metrics

```
cd ./analysis/06_first_order_SHAP_analysis/code/

python3.11 1_first_order_shap_investigator.py --parallelize specialized_global_shap_unsigned_bound

python3.11 1_first_order_shap_investigator.py --parallelize specialized_global_shap_signed_local_SHAP_mean_bound
```

<br>
<br>

# 7 - `Feature Interaction Effect SHAP` Analysis

### NOTES:
* `Feature Interaction Effect SHAP` values include all feature-feature interactions along with the `Individual Feature SHAP` values. 
* Due to the computational demand of calculating `Feature Interaction Effect SHAP` values, we only calculate these values using the best `XGBRegressor` model for each `cell line` and only on the `Unique Binding Patterns`.
* Make sure to update the `XGBOOST_BEST_MODEL_HASHES` variable in the following config files before running the scripts in this section:
    * `./analysis/07_second_order_SHAP_analysis/1_generate_shap_interaction_values/code/1_metadata_variables.yaml`
    * `./analysis/07_second_order_SHAP_analysis/2_analyze_shap_interaction_values/code/1_variable_config.yaml`


## 7.1 - Create table per cell line of all `Unique Binding Patterns` within that cell line

```
cd ./analysis/07_second_order_SHAP_analysis/1_generate_shap_interaction_values/code/

python3.11 2_generate_unique_binding_pattern_ID_table.py
```

## 7.2 - Generate `Feature Interaction Effect SHAP` values 

**Before running the script below, update the `SBATCH_PREFIX` variable.**

```
cd ./analysis/07_second_order_SHAP_analysis/1_generate_shap_interaction_values/code/

python3.11 3_generate_values.py --run_all
```

## 7.3 - Validation of `Feature Interaction Effect SHAP` values

Run the following notebook top to bottom to make sure there are no major technical issues with the generated values from the previous step: 
```
cd ./analysis/07_second_order_SHAP_analysis/1_generate_shap_interaction_values/code/

jupyter notebook 4_validation_of_values_generated.ipynb
```


## 7.4 - Calculate all relevant metric values (variants of canonical `Global SHAP`)

```
cd ./analysis/07_second_order_SHAP_analysis/2_analyze_shap_interaction_values/code/

python3.11 2_second_order_shap_analysis.py --parallelize_all_metric_calcs
```

## 7.5 - Aggregate individual `SHAP` metric values from previous step into 1 table per `SHAP` metric

```
cd ./analysis/07_second_order_SHAP_analysis/2_analyze_shap_interaction_values/code/

python3.11 2_second_order_shap_analysis.py --aggregate_all_metrics
```

## 7.6 - Create necessary derivative datasets that feed into downstream figures for `Feature Interaction Effect SHAP` results

**‼️You must run the notebook cells containing the following functions that are part of the `Jupyter` notebook listed below‼️**
* `retrieve_importance_network()`
* `create_table_from_systematic_screen_of_interactions_for_psi_changes()`
* `find_instances_where_interaction_larger_than_main_effects()`


```
cd ./analysis/07_second_order_SHAP_analysis/2_analyze_shap_interaction_values/code/

jupyter notebook 2_analysis_of_second_order_SHAP_interactions.ipynb
```


