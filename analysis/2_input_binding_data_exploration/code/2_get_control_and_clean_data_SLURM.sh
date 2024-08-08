#!/bin/bash
#SBATCH --account=platiglab    
#SBATCH --partition=largemem    
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --mem=300GB
#SBATCH --array=0-19

# get the input binding data files
files=($(find /project/PlatigLab/data/collaborators/BWH/input_binding_data_and_INCORRECT_SHAP_toy_data_2024-07/input_binding_data/ -iname "*.csv.gz" | sort ))

# run python script where the input is the file name 
python3 ./1_subset_control_and_clean_data.py ${files[$SLURM_ARRAY_TASK_ID]}
