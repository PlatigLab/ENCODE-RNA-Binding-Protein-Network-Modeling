# This script only makes the sweep with normal holdout_r2_scores for the inner folds 
# and the runs from here are used in the next script to calculate outer fold r2 scores

import os

SCRIPT_DIR = "/scratch/jve4pt/platiglib/scripts"
YAML_FILE = "/project/PlatigLab/users/yogi/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/3_choose_dataset_and_model_parameters/wandb_configs/model/3_outer_loop_holdout_r2/final_outer_loop_holdout_r2.yaml"

os.chdir(SCRIPT_DIR)
os.system(
    f'python {SCRIPT_DIR}/send_wandb_sweep.py --memory 128 --cpu 16 --sweep-config {YAML_FILE} --sweep-name "outer_loop_holdout_r2_score"'
)
