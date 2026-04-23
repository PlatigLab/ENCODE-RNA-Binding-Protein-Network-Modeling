#!/bin/bash
#SBATCH --job-name=rbp_pipeline
#SBATCH --output=logs/rbp_%A_%a.out
#SBATCH --error=logs/rbp_%A_%a.err
#SBATCH --account=PlatigLab
#SBATCH --partition=standard
#SBATCH --time=00:60:00
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH --array=0-14

RBPS=(SAFB NOLC1 ZC3H11A EXOSC5 WDR43 RBM5 FXR2 DROSHA STAU2
    CDC40 SRSF7 EIF3H SDAD1 IGF2BP1 AGGF1)

RBP=${RBPS[$SLURM_ARRAY_TASK_ID]}
python3.11 HepG2_Crispr_KD.py $RBP