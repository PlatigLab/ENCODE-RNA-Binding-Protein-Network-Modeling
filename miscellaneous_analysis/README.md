# Miscellaneous Analyses

📂 `01_hg19_ENCODE_pval_correction/`: 

Code related to recalculating p-values for `ENCODE`'s provided data. 


📂 `02_Ayan_hg38_vs_ENCODE_hg19_comparison/`: 

Compare number of total and differential `AS` events between `hg19 ENCODE` and `hg38 Ayan` data.

(e.g. compare `A3SS` for `RBFOX2 KD` in `K562` between both versions of data)


📂 `03_compare_ENCODE_hg19_batch_corrected_vs_uncorrected_PSI/`: 

Understanding whether batch correction affects `PSI` and `deltaPSI` values much. 

This is using `ENCODE`'s `hg19` data that they provide on their portal. 


📂 `04_compare_yogi_and_ayan_datasets/`: 

Compare RBP binding amount and positional preferences between the dataset created by Yogi vs Ayan. 


📂 `05_batch_effect_investigation/`: 

Looking at the effect of sequencing batch on `PSI` values across splice types specifically for `ENCODE` data that was analyzed with `hg38` using Ayan's pipeline. 


📂 `06_ordinal_model_experiment/`:

Mini-experiment to see how well ordinal regression models do as compared to our approach of continuous prediction of `PSI` values.


📂 `07_verify_sample_ordering_from_ayan/`:

Check that correlation between samples that should be in same condition is higher than those that are in different conditions. 
Related to bug/issue I found in Janmejay's code as described here: https://github.com/NNeuralDynamics/as-rmats-turbo-encode/issues/2. 


📂 `08_model_variation_rapid_testing/`:

Run side experiment to see how different combinations of feature and data variations affect the model performance to see if we are missing out on better ways to model this data.


📂 `09_reproduce_probability_SHAP_bug/`:

Add minimal code to reproduce a grave, dangerous, fatal bug in `SHAP` when using `probability` outputs. When asking for too many rows at once, the `local SHAP` values returned are extremely far off from what one would expect (too large numbers). @bcjonescbt has also independently verified this bug. Issue was submitted on GitHub: https://github.com/shap/shap/issues/4151


📂 `10_no_sample_importance_downsampling_SHAP_pilot/`: 

Pilot study to see whether models trained on downsampled data (less 1.0 `PSI` values) are better at modeling differential splicing, in which case we would take these models since the `R^2` values are better. In the end, these models were worse than the lower `R^2` models that we already have. 
