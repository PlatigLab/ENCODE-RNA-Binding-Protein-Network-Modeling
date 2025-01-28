#!/bin/bash

thresholds=(25 50 75 100 150 200 250 500 1000)
cell_lines=(HepG2 K562)
data_path="/project/PlatigLab/data/collaborators/BWH/1_ENCODE_shRNA_RBP_KD_2024-04-hg38-gencode-v29/"

for cell_line in "${cell_lines[@]}"
do

    rbps=($(cut -f4 "../../3_assign_eCLIP_to_splice_junctions/output/bedtools_input/${cell_line}_all_peaks_sorted.bed" | cut -d"_" -f1 | sort | uniq))
    
    for rbp in "${rbps[@]}"
    do

        if [ -n "$(find ${data_path} -iname "${rbp}-*-${cell_line}" | grep -v "Transfection")" ]; then

            for threshold in "${thresholds[@]}"
            do
                
                sbatch --nodes=1 --ntasks=2 --mem=50GB --partition=standard --account=platiglab --output=../SLURM_output/"${cell_line}"_"${rbp}"_"${threshold}".out --error=../SLURM_output/"${cell_line}"_"${rbp}"_"${threshold}".error --wrap="python3 ./1_create_cell-line_rbp_dataset.py --cell_line ${cell_line} --rbp ${rbp} --threshold ${threshold}"

            done
            
        else
            echo "No matching files found for ${cell_line} ${rbp}"

        fi

    done

done
