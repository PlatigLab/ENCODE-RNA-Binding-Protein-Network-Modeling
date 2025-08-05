# Miscellaneous Analyses

📂 `1_hg19_ENCODE_pval_correction/`: 

Code related to recalculating p-values for `ENCODE`'s provided data. 

📂 `2_Ayan_hg38_vs_ENCODE_hg19_comparison/`: 

Compare number of total and differential `AS` events between `hg19 ENCODE` and `hg38 Ayan` data.

(e.g. compare `A3SS` for `RBFOX2 KD` in `K562` between both versions of data)

📂 `3_compare_ENCODE_hg19_batch_corrected_vs_uncorrected_PSI/`: 

Understanding whether batch correction affects `PSI` and `deltaPSI` values much. 

This is using `ENCODE`'s `hg19` data that they provide on their portal. 

📂 `4_compare_yogi_and_ayan_datasets/`: 

Compare RBP binding amount and positional preferences between the dataset created by Yogi vs Ayan. 

📂 `5_batch_effect_investigation/`: 

Looking at the effect of sequencing batch on `PSI` values across splice types specifically for `ENCODE` data that was analyzed with `hg38` using Ayan's pipeline. 

📂 `6_ordinal_model_experiment/`:

Mini-experiment to see how well ordinal regression models do as compared to our approach of continuous prediction of `PSI` values.

📂 `7_verify_sample_ordering_from_ayan/`:

Check that correlation between samples that should be in same condition is higher than those that are in different conditions. 
Related to bug/issue I found in Janmejay's code as described here: https://github.com/NNeuralDynamics/as-rmats-turbo-encode/issues/2. 

📂 `8_reproduce_probability_SHAP_bug/`:

Add minimal code to reproduce a grave, dangerous, fatal bug in `SHAP` when using `probability` outputs. When asking for too many rows at once, the `local SHAP` values returned are extremely far off from what one would expect (too large numbers). @bcjonescbt has also independently verified this bug.
