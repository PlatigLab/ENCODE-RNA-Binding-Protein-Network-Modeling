#!/bin/bash
#SBATCH --job-name=rbp_pipeline
#SBATCH --output=logs/rbp_%A_%a.out
#SBATCH --error=logs/rbp_%A_%a.err
#SBATCH --account=PlatigLab
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH --array=0-51

# -----------------------------------------------------------------------
# RBP lists — keep in sync with precompute_rmats_matches.py
# -----------------------------------------------------------------------
RBPS_HepG2=(SAFB NOLC1 ZC3H11A EXOSC5 WDR43 RBM5 FXR2 DROSHA STAU2
            CDC40 SRSF7 EIF3H SDAD1 IGF2BP1 AGGF1)

RBPS_K562=(NIPBL SAFB NOLC1 ZC3H11A EXOSC5 FXR2 RPS3 ZNF800 
    SDAD1 SRSF7 IGF2BP1 DDX42 MORC2 RYBP DDX21 APEX1
    RPS6 DDX6 GNL3 ELAC2 NPM1 TRA2A METTL1 PRPF8
    ELAVL1 XRCC6 SRSF9 ADAT1 DDX43 RPS11 EIF4E EXOSC10
    RNF187 SF3B1 GARS YWHAG DGCR8)

N_HepG2=${#RBPS_HepG2[@]}   # 15
N_K562=${#RBPS_K562[@]}     # 37
# -----------------------------------------------------------------------
# Map flat array index → (cell_line, RBP)
# Indices 0 .. N_HepG2-1        → HepG2
# Indices N_HepG2 .. N_HepG2+N_K562-1 → K562
# -----------------------------------------------------------------------
IDX=$SLURM_ARRAY_TASK_ID

if [ "$IDX" -lt "$N_HepG2" ]; then
    CELL_LINE="HepG2"
    RBP=${RBPS_HepG2[$IDX]}
else
    CELL_LINE="K562"
    RBP=${RBPS_K562[$((IDX - N_HepG2))]}
fi

mkdir -p logs

echo "Task ${IDX}: ${RBP} in ${CELL_LINE}"
python3.11 CRISPR_KD.py ${RBP} ${CELL_LINE}