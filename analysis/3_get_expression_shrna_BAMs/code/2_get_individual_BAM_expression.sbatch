#!/bin/bash

# run this script on Discovery Northeastern

#SBATCH --partition=express
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --output=../SLURM_output/%A_%a-output.txt
#SBATCH --error=../SLURM_output/%A_%a-error.txt
#SBATCH --mem=90GB
#SBATCH --array=0-459

# get all folders and sort 
encode_ids=($(find /scratch/a.paul/encode-shRNA-KD/ -maxdepth 1 -mindepth 1 -type d | sort))
# choose folder by slurm task ID
encode_id=${encode_ids[$SLURM_ARRAY_TASK_ID]}
# get ENCODE ID  
prefix=$(basename $encode_id)

# run featureCounts command 
# use multiple cores, reverse stranded
# only count paired end reads where both ends were aligned 
# count exonic overlapping reads and summarize at the gene level 
# counts the 2 KD and 2 control BAM files per experiment 
~/subread-2.0.7-Linux-x86_64/bin/featureCounts -T 20 -s 2 -p --countReadPairs -B -t exon -g gene_id -a ~/gencode.v29.primary_assembly.annotation_UCSC_names.gtf.gz --tmpDir /scratch/y.raghav/ -o ../output/$prefix"".txt $encode_id/star/*/*/*.out.bam 
