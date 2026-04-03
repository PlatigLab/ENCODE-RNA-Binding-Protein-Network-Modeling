#!/bin/bash
#SBATCH --job-name=rbp_pipeline
#SBATCH --output=logs/rbp_%A_%a.out
#SBATCH --error=logs/rbp_%A_%a.err
#SBATCH --account=PlatigLab   # allocation name
#SBATCH --partition=interactive  # partition
#SBATCH --time=00:30:00
#SBATCH -n 12
#SBATCH --mem=116G
#SBATCH --array=0-35


RBPS=(
    NIPBL
    SAFB
    NOLC1
    ZC3H11A
    EXOSC5
    FXR2
    RPS3
    ZNF800
    SDAD1
    SRSF7
    IGF2BP1
    DDX42
    MORC2
    RYBP
    DDX21
    APEX1
    RPS6
    DDX6
    GNL3
    ELAC2
    NPM1
    TRA2A
    METTL1
    PRPF8
    ELAVL1
    XRCC6
    SRSF9
    ADAT1
    DDX43
    RPS11
    EIF4E
    EXOSC10
    RNF187
    SF3B1
    GARS
    YWHAG)

RBP=${RBPS[$SLURM_ARRAY_TASK_ID]}

python3.11 K562_Crispr_KD_2.py $RBP
