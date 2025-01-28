#!/bin/bash
#SBATCH --account=platiglab    
#SBATCH --partition=standard    
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --output=../SLURM_output/final_dataset_creation_output_%A_%a.txt
#SBATCH --error=../SLURM_output/final_dataset_creation_error_%A_%a.txt
#SBATCH --mem=50GB
#SBATCH --array=0-17


thresholds=(25 50 75 100 150 200 250 500 1000)
cell_lines=(HepG2 K562)
# data_value_variations=("binary-binding-only" "num-peaks-only" "expression-getmm_no-log_dose-dependent-expression" "expression-getmm_no-log_dose-independent-expression" "expression-getmm_yes-log_dose-dependent-expression" "expression-getmm_yes-log_dose-independent-expression" "expression-tmm_no-log_dose-dependent-expression" "expression-tmm_no-log_dose-independent-expression" "expression-tmm_yes-log_dose-dependent-expression" "expression-tmm_yes-log_dose-independent-expression")


combo=()

for cell_line in "${cell_lines[@]}"
do

    for threshold in "${thresholds[@]}"
    do

        combo+=("${cell_line}_*_${threshold}_*.tsv.gz")

    done

done

# Sort the combo array
sorted_combo=($(printf '%s\n' "${combo[@]}" | sort))

# Assign the sorted combo array back to the original variable
combo=("${sorted_combo[@]}")

# get the index of the combo array for this iteration
find_string="${combo[${SLURM_ARRAY_TASK_ID}]}"
echo $find_string

files=($(find ../output/ -name "${find_string}" -type f | sort))

# Replace asteriks/underscores and provide correct suffix in find_string
output_file=../final_modeling_input_datasets/$(echo $find_string | sed 's/_//' | sed 's/\*//g' | sed 's/.tsv.gz//g')num-peaks-no-kd.tsv

# Ensure the header is consistent across all files
header=$(for file in "${files[@]}"; do zcat "$file" | head -n1; done | sort | uniq)

if [[ $(echo "$header" | wc -l) -ne 1 ]]; then
    echo "Error: Inconsistent headers found in the files."
    exit 1
fi

# Uncompress the first file in the files array
gunzip -c "${files[0]}" | head -n1 > "$output_file"

for file in "${files[@]}" 
do 

    gunzip -c "$file" | tail -n +2 >> "$output_file"

done

# Remove rows containing "chrUn_" or "_random"
sed -i '/chrUn_\|_random/d' "$output_file"

# Remove duplicate rows based on the first column
awk '!seen[$1]++' "$output_file" > "${output_file}_temp"

# Replace the temporary file with the final output file
mv "${output_file}_temp" "$output_file"

gzip "$output_file"
