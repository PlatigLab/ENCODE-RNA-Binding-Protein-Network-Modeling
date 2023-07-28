curl -O -J -L "https://www.encodeproject.org/metadata/?assay_title=eCLIP&files.file_type=bam&type=Experiment&files.processed=true"

mv metadata.tsv eclip_metadata.tsv

curl -O -J -L "https://www.encodeproject.org/metadata/?assay_title=shRNA+RNA-seq&files.file_type=bam&type=Experiment&files.processed=true"

mv metadata.tsv shrna_metadata.tsv