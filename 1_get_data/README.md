### Data Source
Downloaded data and metadata from this URL: 

https://www.encodeproject.org/search/?type=Experiment&assay_title=eCLIP&files.file_type=bed+narrowPeak&biosample_ontology.term_name=HepG2&biosample_ontology.term_name=K562

### Filter Summary 
1. eCLIP assay 
2. narrowPeak BED files
   * These are the final filtered files from ENCODE pipeline. 
3. HepG2 and and K562 cell lines 

### Downloading Data 
`xargs -L 1 curl -O -J -L < ./1_get_data/output/files.txt`

### Metadata
Downloaded as part of above command and can be found here: 
`./1_get_data/output/metadata.tsv`
