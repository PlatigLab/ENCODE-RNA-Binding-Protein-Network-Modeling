#!/bin/bash
#SBATCH --job-name=precompute_rmats
#SBATCH --output=logs/precompute_%a.out   # %a = array index (0 or 1)
#SBATCH --error=logs/precompute_%a.err
#SBATCH --account=PlatigLab
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH -n 20
#SBATCH --mem=160GB
#SBATCH --array=0-1                       # one task per cell line

# Map array index → cell line
CELL_LINES=("HepG2" "K562")
CELL_LINE=${CELL_LINES[$SLURM_ARRAY_TASK_ID]}

echo "Starting precompute for ${CELL_LINE} (task ${SLURM_ARRAY_TASK_ID})"

mkdir -p precompute_logs
python3.11 precompute_rmats_matches.py ${CELL_LINE}