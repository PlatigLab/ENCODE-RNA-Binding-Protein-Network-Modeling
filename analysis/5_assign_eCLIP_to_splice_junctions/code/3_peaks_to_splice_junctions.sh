
for file in ../output/bedtools_input/*.bed; 
do
    echo $file
    /project/PlatigLab/software/bedtools-v2.31.1/bin/bedtools sort -i $file > ../output/bedtools_input/$(basename $file ".bed")_sorted.bed
    rm $file

done


for file in ../output/bedtools_input/*all_peaks_sorted.bed;
do 

    echo $file 

    /project/PlatigLab/software/bedtools-v2.31.1/bin/bedtools closest \
        -a $file \
        -b ../output/bedtools_input/unique_splicing_junctions_sorted.bed \
        -s \
        -d \
        -t first > ../output/peaks_to_splice_junctions/$(basename $file "_all_peaks_sorted.bed")_peaks_to_splice_junctions.bed

done









