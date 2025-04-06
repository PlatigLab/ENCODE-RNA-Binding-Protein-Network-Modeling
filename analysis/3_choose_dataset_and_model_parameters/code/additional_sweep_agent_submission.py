import os
SCRIPT_DIR = "/scratch/jve4pt/platiglib/scripts"
os.chdir(SCRIPT_DIR)

import platiglib.wandb_utils

PROJECT_NAME = "yogi-xgbregressor-hyperparameter-sweep-april-2025"
SWEEP_ID = "ft20ki6n"
NUM_ADDITIONAL_AGENTS = 1
MEMORY = 256
CPUS = 16

wbp = platiglib.wandb_utils.WBProject(PROJECT_NAME)
sweep_mgr = wbp.sweep_manager

sweep_mgr.submit_additional_slurm_agents(
    SWEEP_ID, 
    num_slurm_agents = NUM_ADDITIONAL_AGENTS,
    memory = MEMORY,
    cpus = CPUS,
)
