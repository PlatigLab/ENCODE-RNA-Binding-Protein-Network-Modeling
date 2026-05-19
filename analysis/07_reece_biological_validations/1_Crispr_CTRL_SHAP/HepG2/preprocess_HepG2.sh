#!/bin/bash
#SBATCH --job-name=precompute_bat
#SBATCH --output=logs/precompute.out
#SBATCH --error=logs/precompute.err
#SBATCH --account=PlatigLab
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH -n 4
#SBATCH --mem=64G   # just enough for the BAT

python3.11 HepG2_precompute_rmats_matches.py