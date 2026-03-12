import os
SCRIPT_DIR = "/scratch/jve4pt/platiglib/scripts"
os.chdir(SCRIPT_DIR)

import platiglib.wandb_utils

ACCOUNT="platiglab_paid"
TIME="04:00:00"
PROJECT_NAME = "yogi-linear-models-gencode-v24-v29-matching-exons"
SWEEP_ID = "k2unw4uu"
NUM_ADDITIONAL_AGENTS = 1
MEMORY = 64
CPUS = 16

wbp = platiglib.wandb_utils.WBProject(PROJECT_NAME)
sweep_mgr = wbp.sweep_manager

sweep_mgr.submit_additional_slurm_agents(
    SWEEP_ID, 
    num_slurm_agents = NUM_ADDITIONAL_AGENTS,
    memory = MEMORY,
    cpus = CPUS,
    account=ACCOUNT, 
    time=TIME
)
