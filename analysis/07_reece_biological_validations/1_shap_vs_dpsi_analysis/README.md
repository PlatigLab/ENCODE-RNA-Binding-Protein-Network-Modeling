# SHAP vs dPSI Analysis


# Code

# 1_ignore_old_shap_v_dpsi_individual_RBP_plots

Old one-off plots -> ignore

### 1_rbfox2_plot
- Creates volcano plots for rows where RBFOX2 is bound at pos. 4 and where RBFOX2 is bound at pos. 4 but not 3

### 2_rbfox2_counts_vs_shap
- Histograms of the distribution of SHAP values at position 4 for RBFOX2

### 3_30_volcano_plots
- Creates a volcano plot for RBPs with the highest and lowest mean bound local SHAP values
- Assesses RBP KD events at the position where the RBP is bound (in-silico KD rows)
- Gauges RBP activity (activator / repressor) at a specific position

# 2_figure_3

All code to make Figure 3

## activity_heatmap_and_scatterplot
- Makes heatmap of RBP Activity Score and scatterplot of sign concordance

## Tables
- activity_heatmap.csvs have the RBP activity scores for a 10 event cutoff
- merged_heatmap.csv merges the above csvs

**Key Question:** Do the mean local SHAP values align with the RBP activity demonstrated in the rMATS data?

- Are predicted repressors (-SHAP) actually repressors (-dPSI in in-silico KD rows)?
- Are predicted activators (+SHAP) actually activators (+dPSI in in-silico KD rows)?

**RBP Activity Metric**

- Defined as:`rbp_activity = (num_pos - num_neg) / (num_pos + num_neg)`

where `num_pos` is the number of rMATS significant splicing events (FDR  < 0.05 and dPSI > 0.05) for a given RBP knockdown where that RBP binds at that position (in silico KD rows) with a positive dPSI and `num_neg`  are points with negative dPSI. Basically, (red dots - blue dots) / (red dots + blue dots).

## data_path.txt
- Path to big table

# 3_comparing_event_cutoffs
One-off analysis to compare how RBP activity metric changesdepending on the cuto
