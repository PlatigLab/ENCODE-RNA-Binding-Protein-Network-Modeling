#!/bin/bash

thresholds=(50 100 150 200 250 500)
cell_lines=(HepG2 K562)
data_modes=("all-events")
data_path="/project/PlatigLab/data/collaborators/BWH/1_ENCODE_shRNA_RBP_KD_2024-04-hg38-gencode-v29/"

for cell_line in "${cell_lines[@]}"
do

    rbps=($(sort "../../3_assign_eCLIP_to_splice_junctions/output/eclip_and_shrna_rbps/${cell_line}_shrna_rbps.txt"))

    for rbp in "${rbps[@]}"
    do

        for threshold in "${thresholds[@]}"
        do

            for data_mode in "${data_modes[@]}"
            do  

                output_file="../output/${cell_line}_${rbp}_${threshold}_${data_mode}_num-peaks-no-kd.tsv.gz"
                
                if [ ! -f "${output_file}" ]; then

                    echo $output_file

                    sbatch \
                        --nodes=1 \
                        --ntasks=8 \
                        --mem=64GB \
                        --partition=standard \
                        --account=platiglab \
                        --output=../SLURM_output/"${cell_line}"_"${rbp}"_"${threshold}"_"${data_mode}".out \
                        --error=../SLURM_output/"${cell_line}"_"${rbp}"_"${threshold}"_"${data_mode}".error \
                        --wrap="python3.11 ./1_create_cell-line_rbp_dataset.py --cell_line ${cell_line} --rbp ${rbp} --threshold ${threshold} --data_mode ${data_mode}"
                
                fi

            done

        done

    done

done
