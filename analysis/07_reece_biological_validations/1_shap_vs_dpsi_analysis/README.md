# SHAP vs dPSI Analysis


## Code

## 1_rbfox2_plot
- Creates volcano plots for rows where RBFOX2 is bound at pos. 4 and where RBFOX2 is bound at pos. 4 but not 3

## 2_rbfox2_counts_vs_shap
- Histograms of the distribution of SHAP values at position 4 for RBFOX2

## 3_30_volcano_plots
- Creates a volcano plot for RBPs with the highest and lowest mean bound local SHAP values
- Assesses RBP KD events at the position where the RBP is bound (in-silico KD rows)
- Gauges RBP activity (activator / repressor) at a specific position

## 4_RBP_Activity_Heatmap_Figures
- Makes heatmap of RBP Activity Score:

### RBP Activity Metric

- Based exclusively off the rMATS data
- Meant to be comparable in magnitude to SHAP values (positive, negative)

**Key Question:** Do the mean local SHAP values align with the RBP activity demonstrated in the rMATS data?

- Are predicted repressors (-SHAP) actually repressors (-dPSI in in-silico KD rows)?
- Are predicted activators (+SHAP) actually activators (+dPSI in in-silico KD rows)?

**RBP Activity Metric**

- Defined as:`rbp_activity = (num_pos - num_neg) / (num_pos + num_neg)`

where `num_pos` is the number of rMATS significant splicing events (FDR  < 0.05 and dPSI > 0.05) for a given RBP knockdown where that RBP binds at that position (in silico KD rows) with a positive dPSI and `num_neg`  are points with negative dPSI. Basically, (red dots - blue dots) / (red dots + blue dots).

## data_path.txt
- Path to big table