import os
SCRIPT_DIR = "/scratch/jve4pt/platiglib/scripts"
os.chdir(SCRIPT_DIR)

import platiglib.wandb_utils

PROJECT_NAME = "yogi-sample-importance-fall-2025"
SWEEP_ID = "3wsjjtdx"
NUM_ADDITIONAL_AGENTS = 500
MEMORY = 128
CPUS = 16

wbp = platiglib.wandb_utils.WBProject(PROJECT_NAME)
sweep_mgr = wbp.sweep_manager

sweep_mgr.submit_additional_slurm_agents(
    SWEEP_ID, 
    num_slurm_agents = NUM_ADDITIONAL_AGENTS,
    memory = MEMORY,
    cpus = CPUS,
)
