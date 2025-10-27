# HACKY!

# This script copies over the model, explainer, and SHAP data 
# multiple times because the SHAP Network Analyzer expects 5 items per cell line
# but we only have 1 per cell line in this quick & dirty side experiment.

# Hence, we just duplicate the data 5 times for each cell line.

import os, shutil, glob, gzip, json, pickle
import pandas as pd

json_dir = "/project/PlatigLab/users/yogi/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/03_choose_dataset_and_model_parameters/output/model_reproduction/model_parameters"
model_dir = "/project/PlatigLab/users/yogi/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/04_run_final_models_and_SHAP/outputs/pickled_models/XGBRegressor"
explainer_dir = "/project/PlatigLab/users/yogi/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/04_run_final_models_and_SHAP/outputs/SHAP/regular/normal/explainer_objects"
shap_dir = "/project/PlatigLab/users/yogi/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/04_run_final_models_and_SHAP/outputs/SHAP/regular/normal/shap_values"

json_files = sorted(glob.glob(f"{json_dir}/*.json"))

for json_file in json_files:
    hash_prefix = json_file.split("/")[-1].split(".json")[0]

    # Try to infer cell line from json file contents or filename
    with open(json_file, "r") as f:
        content = f.read()
        cell_line = json.loads(content).get("cell_line")

    # Copy model pickle as gzip
    model_file = f"./1_{cell_line}.pkl"
    with open(model_file, "rb") as f_in:
        model_obj = pickle.load(f_in)
        with gzip.open(f"{model_dir}/{hash_prefix}.pkl.gz", "wb") as f_out:
            pickle.dump(model_obj, f_out)

    # Copy explainer pickle
    explainer_file = f"./3_{cell_line}_SHAP_explainer.pkl"
    shutil.copy2(
        explainer_file, 
        f"{explainer_dir}/{hash_prefix}.pkl"
    )

    # Copy SHAP feather file 
    shap_file = f"./3_{cell_line}_SHAP.feather"
    shutil.copy2(
        shap_file, 
        f"{shap_dir}/{hash_prefix}.feather"
    )

    print(f"Copied data for {cell_line} with hash: {hash_prefix}")