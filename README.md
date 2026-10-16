# Explainable machine learning reveals an `RBP` regulatory logic of `exon skipping`

## TL;DR: 

Elucidating the **spatial** and **spatial + combinatorial** splicing roles of RBPs with a **first-ever approach** that simultaneously: 

* Takes the entire compendium of `RBP binding data` (`eCLIP`) and matched `shRNA KD RNA-Seq` data across 2 cell lines, with the aim of predicting the `Percent Spliced In (PSI)` decimal value of skipped exon events across the transcriptome.
* Is not model-prediction-centric but rather interpretability-centric (using `SHAP`) to uncover putative novel regulatory rules.

## Figure/Table Creation

All figures/tables can be re-created using [this Jupyter notebook](./FIGURE_AND_TABLE_CREATION.ipynb).

## Reproduction (End-to-End)

Please see the [analysis reproduction document](./REPRODUCTION.md). 

## Citation

Yogindra Raghav, Ayan Paul, Reece Anderson, Shalini Karthyk, Annette Iturralde, Janmejay Vyas, Jennifer Dy, Brett C. Jones, Peter J. Castaldi, John Platig. *Explainable machine learning reveals an RBP regulatory logic of exon skipping*. **bioRxiv** (2026). doi: https://doi.org/10.64898/2026.05.29.728731

## Folder Structure 

📂`analysis/`: 

Contains the main analyses in the sequential order done that went towards the publication. 

📂`inputs/`: 

Contains general input data that is useful across the entire repository. (e.g. `ENCODE metadata` info). 

📂`miscellaneous_analysis/`:

Contains analyses that were not part of the publication but answered highly important and highly specific questions.  

