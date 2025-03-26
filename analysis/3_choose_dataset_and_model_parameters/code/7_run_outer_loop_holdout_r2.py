import platiglib.wandb_utils
import os

SCRIPT_DIR = "/scratch/jve4pt/platiglib/scripts"
os.chdir(SCRIPT_DIR)

project = "yogi-xgbregressor-hyperparameter-sweep-march-2025"
sweep_id = "f1661ffh"

outer_loop_holdout_r2_sweep_ids = platiglib.wandb_utils.WBProject(
        project
    ).sweep_manager.get_runs_in_sweep(
        sweep_id
    )

for run_id in outer_loop_holdout_r2_sweep_ids:
    os.system(
        f"sbatch -J {run_id}_outer_loop_holdout_r2 run_python_script.slurm run_outer_loop_test.py --project {project} --run {run_id}"
    )

