#!/bin/bash

thresholds=(50 100 500 1000 2000 5000 10000)
cell_lines=(HepG2 K562)
data_value_variations=("binary-binding-only" "num-peaks-only" "expression-getmm_no-log_dose-dependent-expression" "expression-getmm_no-log_dose-independent-expression" "expression-getmm_yes-log_dose-dependent-expression" "expression-getmm_yes-log_dose-independent-expression" "expression-tmm_no-log_dose-dependent-expression" "expression-tmm_no-log_dose-independent-expression" "expression-tmm_yes-log_dose-dependent-expression" "expression-tmm_yes-log_dose-independent-expression")


# # for each cell line
# for cell_line in "${cell_lines[@]}"
# do 

#     # Get a list of all gzip'd files matching the cell line
#     files=($(find ../output/ -type f -name "${cell_line}_*.gz"))

#     # Check if there are any files
#     if [ ${#files[@]} -eq 0 ]; then
#         echo "No gzip'd files found in the directory."
#         exit 1
#     fi

#     # Uncompress and check the first line of the first file
#     first_file="${files[0]}"
#     uncompressed_output=$(gunzip -c "$first_file" | head -n 1)

#     # Loop through the remaining files and compare the first line
#     for file in "${files[@]:1}"; do
#         uncompressed_line=$(gunzip -c "$file" | head -n 1)
#         if [ "$uncompressed_line" != "$uncompressed_output" ]; then
#             echo "The first line of $file is different from the first line of $first_file."
#             exit 1
#         fi
#     done

# done

# echo "The first line of all uncompressed files is the same."


for cell_line in "${cell_lines[@]}"
do

    for threshold in "${thresholds[@]}"
    do

        for data_value_variation in "${data_value_variations[@]}"
        do

            array=($(find ../output/ -type f  -name "${cell_line}_*_${threshold}_${data_value_variation}.tsv.gz"))

            output_file="../final_modeling_input_datasets/${cell_line}_${threshold}_${data_value_variation}_output.txt"
            echo "$output_file"
            # touch "$output_file"


            # sbatch --nodes=1 --ntasks=1 --mem=50GB --partition=standard --account=platiglab --output=../SLURM_output/"${cell_line}"_"${rbp}"_"${threshold}".out --error=../SLURM_output/"${cell_line}"_"${rbp}"_"${threshold}".error --wrap="python3 1_script.py --cell_line ${cell_line} --rbp ${rbp} --threshold ${threshold}"
    
        done 

    done

done
