#!/bin/bash
#SBATCH --job-name=rbp_pipeline
#SBATCH --output=logs/rbp_%A_%a.out
#SBATCH --error=logs/rbp_%A_%a.err
#SBATCH --account=PlatigLab   # allocation name
#SBATCH --partition=interactive  # partition
#SBATCH --time=00:30:00
#SBATCH -n 12
#SBATCH --mem=116
#SBATCH --array=0-0


RBPS=(
    NIPBL
)

RBP=${RBPS[$SLURM_ARRAY_TASK_ID]}

python3.11 K562_Crispr_KD.py $RBP