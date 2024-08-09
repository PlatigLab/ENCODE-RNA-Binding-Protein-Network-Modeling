# run this script on Discovery NorthEastern

# get subread v2.0.7, untar, and then get binary for featurecounts
wget "https://sourceforge.net/projects/subread/files/subread-2.0.7/subread-2.0.7-Linux-x86_64.tar.gz/download" -O subread-2.0.7.tar.gz
tar -zxvf subread-2.0.7.tar.gz 

# get the gencode v29 GTF file used throughout the analysis
wget "https://www.encodeproject.org/files/gencode.v29.primary_assembly.annotation_UCSC_names/@@download/gencode.v29.primary_assembly.annotation_UCSC_names.gtf.gz" -O gencode.v29.primary_assembly.annotation_UCSC_names.gtf.gz


