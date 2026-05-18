## how to use pipeline

1. Pre-process the BATs

- Use preprocess_HepG2.sh to run the .py file that preprocesses the BAT
- Edit the script to include all RBPs of interest (those with Crispr KD, eCLIP, NO sHRNA)

2. Use the CL_Slurm script to actually run the pipeline

- It calls the Crispr_KD script