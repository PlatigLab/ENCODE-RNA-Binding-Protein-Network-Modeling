wget "https://sourceforge.net/projects/subread/files/subread-2.0.7/subread-2.0.7-Linux-x86_64.tar.gz/download" -O subread-2.0.7.tar.gz

tar -zxvf subread-2.0.7.tar.gz 

wget "https://www.encodeproject.org/files/gencode.v29.primary_assembly.annotation_UCSC_names/@@download/gencode.v29.primary_assembly.annotation_UCSC_names.gtf.gz" -O gencode.v29.primary_assembly.annotation_UCSC_names.gtf.gz


featureCounts -p --countReadPairs -B -t exon -g gene_id -a annotation.gtf -o counts.txt mapping_results_PE.bam
