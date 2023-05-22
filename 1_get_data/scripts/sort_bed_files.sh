#!/bin/bash

files=$(find /project/PlatigLab/RBP/encode/filtered_peaks/raw_unsorted/ -type f)

for file in ${files[@]}
do 
    new_file=$(basename $file .bed.gz).sorted.bed
    /home/jve4pt/.yogi_utils/bedtools-2.30.0/bedtools sort -i $file > /project/PlatigLab/RBP/encode/filtered_peaks/sorted/$new_file
done