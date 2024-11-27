distances=(25 50 75 100 125 150 175 200 225 250 500 1000 2000 5000 10000)


for file in ../output/bedtools_input/*.bed; 
do
    echo $file
    /project/PlatigLab/software/bedtools-v2.31.1/bin/bedtools sort -i $file > ../output/bedtools_input/$(basename $file ".bed")_sorted.bed
    rm $file

done


for distance in ${distances[@]};
do 

    for file in ../output/bedtools_input/*all_peaks_sorted.bed;
    do 

        echo $file 

        output_file = ../output/peaks_to_splice_junctions/$(basename $file "_all_peaks_sorted.bed")_''$distance''_peaks_to_splice_junctions.bed

        /project/PlatigLab/software/bedtools-v2.31.1/bin/bedtools window \
            -a $file \
            -b ../output/bedtools_input/unique_splicing_junctions_sorted.bed \
            -w $distance \
            -sm > $output_file

        gzip $output_file

    done
done
