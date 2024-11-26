#!/bin/bash

thresholds=(25 50 75 100 125 150 175 200 225 250 500 1000)
cell_lines=(HepG2 K562)

for cell_line in "${cell_lines[@]}"
do

    rbps=($(cut -f4 /project/PlatigLab/users/yogi/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/3_create_RBP_ML_input/3_assign_eCLIP_to_splice_junctions/output/bedtools_input/${cell_line}_all_peaks_sorted.bed | cut -d"_" -f1 | sort | uniq))
    
    for rbp in "${rbps[@]}"
    do

        for threshold in "${thresholds[@]}"
        do

            sbatch --nodes=1 --ntasks=1 --mem=50GB --partition=standard --account=platiglab --output=../SLURM_output/"${cell_line}"_"${rbp}"_"${threshold}".out --error=../SLURM_output/"${cell_line}"_"${rbp}"_"${threshold}".error --wrap="python3 ./1_create_cell-line_rbp_dataset.py --cell_line ${cell_line} --rbp ${rbp} --threshold ${threshold}"
        
        done

    done

done
