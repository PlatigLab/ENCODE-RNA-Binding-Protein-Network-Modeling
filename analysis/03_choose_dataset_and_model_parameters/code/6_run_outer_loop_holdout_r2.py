import os

SCRIPT_DIR = "/scratch/jve4pt/platiglib/scripts"

PROJECTS = {
    "XGBoost": {
            "project": "yogi-rbp-ml-gencode-v24-v29-matching-exons", 
            "id_file": "../output/chosen_models_for_outer_loop/outer_loop_holdout_run_ids.txt"
        }, 
    "ElasticNet": {
            "project": "yogi-linear-models-gencode-v24-v29-matching-exons", 
            "id_file": "../output/chosen_models_for_outer_loop/elasticnet_run_ids.txt"
        }
}

for model in PROJECTS:
    PROJECT  = PROJECTS[model]['project']
    OUTPUT_FILE = PROJECTS[model]['id_file']

    with open(OUTPUT_FILE, "r") as f:
        run_ids = [line.strip() for line in f]
    
    PROJECTS[model]['ids'] = run_ids
    
    
os.chdir(SCRIPT_DIR)

for model in PROJECTS:

    PROJECT = PROJECTS[model]['project']
    run_ids = PROJECTS[model]['ids']

    for run_id in run_ids:

        os.system(
            f"sbatch -J {run_id}_outer_loop_holdout_r2 run_python_script.slurm run_outer_loop_test.py --project {PROJECT} --run {run_id}"
        )

