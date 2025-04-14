import os

SCRIPT_DIR = "/scratch/jve4pt/platiglib/scripts"
OUTPUT_FILE = "../output/chosen_models_for_outer_loop/elasticnet_run_ids.txt"
PROJECT = "yogi-RBP-ML-linear-models-april-2025"

run_ids = []
with open(OUTPUT_FILE, "r") as f:
    for line in f:
        run_ids.append(line.strip())

os.chdir(SCRIPT_DIR)
for run_id in run_ids:
    os.system(
        f"sbatch -J {run_id}_outer_loop_holdout_r2 run_python_script.slurm run_outer_loop_test.py --project {PROJECT} --run {run_id}"
    )

