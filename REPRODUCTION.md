
<h1 style="color: teal; font-weight: bold; font-size: 30px;">All commands here assume you are in the root directory of this project.</h1>

# 1 - Data Creation
## 1.1 -  Assign `eCLIP` peaks to splice junctions 

‼️**IMPORTANT: the below assumes you already have a folder of `rMATS` results and a folder with the `ENCODE eCLIP peaks`.**‼️

Before runnning the script below, set the `SE_glob` variable to where **all SE files** are within **each RBP's subfolder** before running:

```
python3.11 ./analysis/01_create_RBP_ML_input/3_assign_eCLIP_to_splice_junctions/code/eclip_assigner.py
```

## 1.2 - Create `binding graphs` per each unique combination of `Cell Line` & `RBP` at various distance thresholds
#### NOTE: we use the `100` distance threshold for this entire project.

Before running below script, set `rmats_data_path` to the same path as `SE_glob` variable in the previous script. 

```
bash ./analysis/01_create_RBP_ML_input/4_create_num_peaks_ML_input/code/2_submission_script.sh
```

## 1.3 -  Concatenate `binding graphs` across all RBPs per each `cell line` and distance threshold
#### NOTE: this script removes duplicate `binding graphs` that are caused by `control shRNA RNA-Seq` samples being used as controls for multiple `RBP KD RNA-Seq` samples.

```
sbatch ./analysis/01_create_RBP_ML_input/4_create_num_peaks_ML_input/code/3_concatenate_results.sh
```

## 1.4 -  Run data QC assertions and assign `exon IDs` to each `binding graph`
#### NOTE: this does not do the `in-silico KD` procedure as that happens later on.

* Before running below script, set the following variables: 
    * `OUTPUT_DIR`
    * `DATA_VERSION`
    * `DATA_PATH`
```
python3.11 ./analysis/01_create_RBP_ML_input/4_create_num_peaks_ML_input/code/4_exon_assignment_and_data_assertions.py
```

## 1.5 - Get `RBP KD` information for each `binding graph` at each `position`
#### NOTE: For each `binding graph` from an `RBP KD RNA-Seq` sample, we see if the `KD RBP` is originally bound in the `eCLIP` data at any of the `6 positions` and set `True` if so and `False` otherwise with `Control shRNA RNA-Seq` samples always set to `False`. 

Before running below script, set `DATA_PATH` to the same path as the `DATA_PATH` variable from the script in the previous step.

```
python3.11 ./analysis/01_create_RBP_ML_input/4_create_num_peaks_ML_input/code/5_RBP_KD_statistics.py
```

<br>
<br>

# 2 - Pre-Modeling Data Exploration
## 2.1 - Across both `cell lines`, explore `PSI` distributions and `RBP Binding` preference/frequency

* Set `DATA_PATH` variable in the following file to be the same path as the `DATA_PATH` variable from the script in the previous step: 
    * `./analysis/02_input_binding_exploration/code/1_binding_analyzer.py`

‼️**IMPORTANT: the notebook here automatically chooses the "distance to splice junction" threshold as 100 and only keeps rows with 40 or more sequencing reads (i.e. the PSI calculation came from having at least 40 reads)**‼️

```
Run notebook from top to bottom: 

./analysis/02_input_binding_exploration/code/1_yogi_binding_analyzer.ipynb
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

* Analyses in this section that require `platiglib` MUST be run within the virtual environment setup from above. 
* Set the scripts directory (the `scripts/` folder within `platiglib`) for the many `Python` files mentioned in this section.
* Make sure the sweep configuration `YAML` files passed in have `name` and `project` under the `sweep` top-key properly set. 👇
```
sweep:
  name: <SWEEP_NAME>
  project: <WANDB_PROJECT_NAME>

{rest of YAML file...}
```

## 3.2 - Create `platiglib_cache` data cache before model training

```
python3.11 ./analysis/03_choose_dataset_and_model_parameters/code/1_submit_data_caching_sweep.py
```

## 3.3 - Hyperparameter Tuning

### 3.3.1 - Sequential, interrelated hyperparameter tuning strategy (`XGBRegressor`)

#### NOTE: the idea here is to make hyperparameter tuning more feasible by grouping hyperparameters together that most influence each other and organizing these based on which groups of hyperparameters are most to least important (e.g. we tune a set of hyperparameters and then fix them to the best values before tuning the next set of hyperparameters)

#### 3.3.1.1 - 1st set: `learning_rate`, `max_depth`, `early_stopping_rounds`
NOTE: it is not common to sweep `early_stopping_rounds` but for our own exploration we did so. **ALL subsequent sweeps used `early_stopping_rounds=100` (a canonical value)**

```
python3.11 ./analysis/03_choose_dataset_and_model_parameters/code/2_tree_construction_parameters.py
```

#### 3.3.1.2 - 2nd set: `min_child_weight`, `gamma`

##### NOTE: From previous sweep, take best performing swept hyperparameters by average `holdout_r2_score` across the seeds and per `cell line` to include here: 


* `HepG2`: 
    * `./analysis/03_choose_dataset_and_model_parameters/wandb_configs/model/2_sequential_sweeps/2_tree_splitting_parameters_HepG2.yaml`
* `K562`: 
    * `./analysis/03_choose_dataset_and_model_parameters/wandb_configs/model/2_sequential_sweeps/2_tree_splitting_parameters_K562.yaml`

After, you can run: 

```
python3.11 ./analysis/03_choose_dataset_and_model_parameters/code/3_tree_splitting_parameters.py
```        

#### 3.3.1.3 - 3rd set: `colsample_bytree`, `subsample`

##### NOTE: From previous sweep, take best performing swept hyperparameters by average `holdout_r2_score` across the seeds and per `cell line` to include here:

* `HepG2`: 
    * `./analysis/03_choose_dataset_and_model_parameters/wandb_configs/model/2_sequential_sweeps/3_data_sampling_HepG2.yaml`
* `K562`: 
    * `./analysis/03_choose_dataset_and_model_parameters/wandb_configs/model/2_sequential_sweeps/3_data_sampling_K562.yaml`

After, you can run:

```
python3.11 ./analysis/03_choose_dataset_and_model_parameters/code/4_data_sampling.py
```

#### 3.3.1.4 - 4th set: `reg_alpha`, `reg_lambda`

##### NOTE: From previous sweep, take best performing swept hyperparameters by average `holdout_r2_score` across the seeds and per `cell line` to include here:

* `HepG2`: 
    * `./analysis/03_choose_dataset_and_model_parameters/wandb_configs/model/2_sequential_sweeps/4_regularization_HepG2.yaml`
* `K562`:
    * `./analysis/03_choose_dataset_and_model_parameters/wandb_configs/model/2_sequential_sweeps/4_regularization_K562.yaml`

After, you can run:
```
python3.11 ./analysis/03_choose_dataset_and_model_parameters/code/5_regularization.py
```

### 3.3.2 - `ElasticNet` hyperparameter tuning

#### NOTE: We use a different `W&B` project for linear models as the results summary table has a different schema than the table for `XGBRegressor` models. 

```
python3.11 ./analysis/03_choose_dataset_and_model_parameters/code/7_run_elasticnet.py
```

## 3.4 - Hyperparameter Tuning Results Exploration

#### NOTE: make sure to set the `W&B Project` names within the `sweep_projects` variable of the analysis class file here: `./analysis/03_choose_dataset_and_model_parameters/code/dataset_and_model_parameter_chooser.py`

You can then run the following notebook top to bottom: 
`./analysis/03_choose_dataset_and_model_parameters/code/choosing_parameters.ipynb`

‼️**To continue to the next section, you must make sure to execute code cells that contain the following functions**‼️

* `get_run_ids_for_outer_loop_holdout_r2_scores()`
* `get_elasticnet_run_ids()`

## 3.5 - Retrieve model performance on data never used during tuning (`final holdout dataset`)

#### NOTE: make sure to update the `PROJECTS` dictionary in the below script and specifically the `project` sub-keys to the correct `W&B` project names. 

After, you can run the following: 

```
python3.11 ./analysis/03_choose_dataset_and_model_parameters/code/6_run_outer_loop_holdout_r2.py

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

# 4 - Final Model Training & `First Order SHAP` Value Generation
## 4.1 - Train top 5 models for `XGBRegressor` and `ElasticNet` for both `HepG2` and `K562` (20 models total)

### 4.1.1 - Create data splits for final model training and evaluation

The following script creates the `train`, `validation`, and `test` splits for the final model training and evaluation for each cell line. 

The purpose of this is to make sure that the same data splits are used across the different models trained in the next step. 

```
python3.11 ./analysis/04_run_final_models_and_SHAP/code/1_replicate_platiglib_model.py --split_data
```

### 4.1.2 - Train final models for each `cell line` and model type

```
python3.11 ./analysis/04_run_final_models_and_SHAP/code/1_replicate_platiglib_model.py --run_all
```

### 4.1.3 - Generate `First Order SHAP` values for each of the final models trained in the previous step

#### NOTE 1: `First Order` refers to the fact that we are only generating `SHAP` values for the original features (each `RBP` binding at each `position`) and not generating `SHAP` values for the feature-feature interaction terms (i.e. `Second Order SHAP` values).

#### NOTE 2: `SHAP` values are generated for the `XGBRegressor` models using `tree_path_dependent` algorithm with the units being in `log-odds` space

#### NOTE 3: `SHAP` values are generated for all the `unique binding patterns` (i.e. the set of all unique combinations of `RBP binding` across the `6 positions`)

```
python3.11 ./analysis/04_run_final_models_and_SHAP/code/2_run_SHAP_analysis.py --run_all_normal

```

### 4.1.4 - Join `SHAP` values for `unique binding patterns` with original data

**This step takes the `SHAP` values for the `unique binding patterns` generated in the previous step and joins them with the original dataset with all the `binding graphs`.**

```
python3.11 ./analysis/04_run_final_models_and_SHAP/code/3_join_unique_binding_SHAP_results_with_all_data.py --run_all normal
```

<br>
<br>

# 5 - Assign `PPI` Status for each 2-Way Combination of `eCLIP` RBPs in either `cell line`

This resource table is useful for downstream analyses related to "known" `RBP PPIs`. 

```
python3.11 ./analysis/05_generate_eCLIP_RBP_PPIs/retrieve_ppis.py
```

<br>
<br>

# 6 - `First Order SHAP` Analysis

## 6.1 - Run data QC and data cache creation to prepare for downstream analyses

### 6.1.1 - Create data caches

👇 invokes `load_final_SHAP_data(underlying_data="All-Data")`
```
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
python3.11 1_first_order_shap_investigator.py --parallelize create_test_data_dpsi_vs_local_SHAP_scatterplot_data
```
**Creates

### 6.1.2 - Run data QC assertions 

👇 invokes `run_data_quality_assertions()`
```
python3.11 1_first_order_shap_investigator.py --parallelize data_qc_assertions
```

👇 invokes `check_no_SHAP_variance_per_binding_pattern()`
```
python3.11 1_first_order_shap_investigator.py --parallelize no_shap_variance_per_binding_pattern
```

👇 invokes `assert_SHAP_additivity()`
```
python3.11 1_first_order_shap_investigator.py --parallelize shap_additivity_assertions
```

### 6.1.3 - Run notebook of a large and diverse number of analyses related to the `First Order SHAP` values

Run the following notebook with the specific sections to be run included after. 

The following sections of the notebook are either those that: 
* Were directly part of our publication
* Would be very important to someone delving deeper into this data and problem but were not included in our publication.

`./analysis/06_first_order_SHAP_analysis/code/2_first_order_shap_analysis_results.ipynb`

#### Relevant sections (as denoted with headers in <span style="color: teal;">teal</span> color within notebook): 
* Data QC + Cache Creation
* Predicted vs Actual Plots 
* Global SHAP Across Models and for Average Local SHAP
* Position 3 + 4 Significance in Global SHAP/ElasticNet
* Is Position 3/4 Activator-Like and Other Positions Repressor-Like?
* Global SHAP across Binding Modes & Positional Vignettes
* Local SHAP Sign Analysis
* CLASSIFICATION VIEW: Differential PSI and Local SHAP

<br>
<br>

# 7 - `Second Order SHAP` Analysis

### NOTE 1: `Second Order SHAP` values include all feature-feature interactions along with the `First Order SHAP` values (i.e. main effects)

### NOTE 2: due to the computational demand of calculating `Second Order SHAP` values, we only calculate these values using the best `XGBRegressor` model for each `cell line` and only on the `unique binding patterns`

### NOTE 3: make sure to update the `XGBOOST_BEST_MODEL_HASHES` variable in the following config files before running the scripts in this section:
* `./analysis/07_second_order_SHAP_analysis/1_generate_shap_interaction_values/code/1_metadata_variables.yaml`
* `./analysis/07_second_order_SHAP_analysis/2_analyze_shap_interaction_values/code/1_variable_config.yaml`

<br>


## 7.1 - Create table per cell line of all `unique binding patterns` within that cell line

```
python3.11 ./analysis/07_second_order_SHAP_analysis/1_generate_shap_interaction_values/code/2_generate_unique_binding_pattern_ID_table.py
```

## 7.2 - Generate `Second Order SHAP` values 

```
python3.11 ./analysis/07_second_order_SHAP_analysis/1_generate_shap_interaction_values/code/3_generate_values.py --run_all
```

## 7.3 - Validation of `Second Order SHAP` values

Run the following notebook top to bottom to make sure there are no major technical issues with the generated values from the previous step: 
`./analysis/07_second_order_SHAP_analysis/1_generate_shap_interaction_values/code/4_validation_of_values_generated.ipynb`


## 7.4 - Calculate all relevant metric values (variants of canonical `Global SHAP`)

```
python3.11 2_second_order_shap_analysis.py --parallelize_all_metric_calcs
```

## 7.5 - Aggregate metric values from previous step

```
python3.11 2_second_order_shap_analysis.py --aggregate_all_metrics
```

## 7.6 - Run notebook to explore the `Second Order SHAP` analysis results

Run the following notebook top to bottom to explore the results from the `Second Order SHAP` analysis: 
`./analysis/07_second_order_SHAP_analysis/2_analyze_shap_interaction_values/code/2_analysis_of_second_order_SHAP_interactions.ipynb`


## TODO make sure that it's still necessary to invert `dPSI` values by checking raw PSI values
