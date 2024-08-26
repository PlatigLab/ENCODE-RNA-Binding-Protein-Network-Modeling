#!/bin/bash

thresholds=(50 100 500 1000 2000 5000 10000)
cell_lines=(HepG2 K562)

for cell_line in "${cell_lines[@]}"
do

    rbps=($(cut -f4 /project/PlatigLab/users/yogi/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/5_assign_eCLIP_to_splice_junctions/output/bedtools_input/${cell_line}_all_peaks_sorted.bed | cut -d"_" -f1 | sort | uniq))
    
    for rbp in "${rbps[@]}"
    do

        for threshold in "${thresholds[@]}"
        do

            sbatch --nodes=1 --ntasks=1 --mem=50GB --partition=standard --account=platiglab --output=../SLURM_output/"${cell_line}"_"${rbp}"_"${threshold}".out --error=../SLURM_output/"${cell_line}"_"${rbp}"_"${threshold}".error --wrap="python3 1_script.py --cell_line ${cell_line} --rbp ${rbp} --threshold ${threshold}"
        
        done

    done

done
