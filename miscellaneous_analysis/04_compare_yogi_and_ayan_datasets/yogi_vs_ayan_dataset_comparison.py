import glob
import polars as pl, seaborn as sns, matplotlib.pyplot as plt, pandas as pd
from dataclasses import dataclass
from loguru import logger


@dataclass
class DatasetComparer:
    distance: int = None

    cell_lines = ["K562", "HepG2"]
    
    # dictionary to rename the splice junction positions to numbers
    splice_junction_position_renaming = {
        "5_left": 1, 
        "5_right": 2, 
        "center_left": 3, 
        "center_right": 4, 
        "3_left": 5, 
        "3_right": 6
    }
    
    def __post_init__(self):
        self.load_graphs()

    
    def load_graphs(self): 

        yogi_data_dir = "/project/PlatigLab/data/RBP_ML/2_yogi_dataset_RBP_ML_data_november_2024/"
        ayan_data_dir = "/project/PlatigLab/data/collaborators/BWH/6_ols_regression_and_xgbdt_models_2024_11/bdt-xgb-models-2024-11/"
        datasets = {
            "yogi": {},
            "ayan": {}
        }

        for cell_line in self.cell_lines:

            logger.info(f"Loading data for {cell_line}...")

            ayan_data = pl.scan_csv(
                f"{ayan_data_dir}/*-{cell_line}-{self.distance}-*/{cell_line}-{self.distance}-50-50-shap-all-data.dat", 
                separator=",",
                glob=True, 
            ).select(pl.exclude("^.*_shap$")).collect(streaming=True)
            ayan_data = ayan_data.rename(
                {
                    col: col.split('_')[0] + f"_{self.splice_junction_position_renaming['_'.join(col.split('_')[1:])]}" + "_binding" for col in ayan_data.columns if col.endswith(("_left", "_right"))
                }
            )

            yogi_data = pl.scan_csv(
                f"{yogi_data_dir}/{cell_line}_{self.distance}_num-peaks-no-kd.tsv.gz",
                separator="\t",
            ).select(pl.col("*")).collect(streaming=True)

            datasets["yogi"][cell_line] = yogi_data
            datasets["ayan"][cell_line] = ayan_data

        self.datasets = datasets
        logger.success("Data loaded successfully.")

    
    def subset_to_transcriptome(self): 

        transcriptome_only = {
            "yogi": {},
            "ayan": {}
        }

        for cell_line in self.cell_lines: 

            ayan_data = self.datasets['ayan'][cell_line].filter(
                pl.col("RBP_KD") == "NONE"
            ).select(
                pl.col("^.*_binding$"), pl.col("^ENSE.*$")
            )
            logger.info(f'# unique 3-exon combinations in Ayan {cell_line} data: {ayan_data.select(pl.col("^ENSE.*$")).unique().shape[0]}')
            assert ayan_data.null_count().sum().sum_horizontal().item()==0, f"Missing or null values found in ayan_data for {cell_line}"     

            transcriptome_only["ayan"][cell_line] = ayan_data.unique().drop(pl.col("^ENSE.*$"))
            logger.info(f"Subsetted Ayan data to transcriptome for {cell_line}. Shape: {transcriptome_only['ayan'][cell_line].shape}")
        
            yogi_data = self.datasets['yogi'][cell_line].filter(
                pl.col("RBP_KD_Target") == "CTRL"
            ).select(
                pl.col("^.*_binding$"), pl.col("^.*Exon$")
            )
            logger.info(f'# unique 3-exon combinations in Yogi {cell_line} data: {yogi_data.select(pl.col("^.*Exon$")).unique().shape[0]}')
            assert yogi_data.null_count().sum().sum_horizontal().item()==0, f"Missing or null values found in yogi_data for {cell_line}"

            transcriptome_only["yogi"][cell_line] = yogi_data.unique().drop(pl.col("^.*Exon$"))
            logger.info(f"Subsetted Yogi data to transcriptome for {cell_line}. Shape: {transcriptome_only['yogi'][cell_line].shape}")

        self.transcriptome_only = transcriptome_only
        logger.success("Data subsetted to transcriptome successfully")


    def plot_amount_binding(self, column_normalized=None, transcriptome_only=None): 
        
        assert column_normalized is not None and transcriptome_only is not None, "Please specify column_normalized and transcriptome_only parameters"

        for cell_line in self.cell_lines:

            heatmap_data = {}
            for dataset in ["yogi", "ayan"]:
                heatmap_data[dataset] = {}

                if transcriptome_only:
                    data = self.transcriptome_only[dataset][cell_line]
                else: 
                    data = self.datasets[dataset][cell_line]

                    if dataset == "yogi":
                        data = data.filter(pl.col("RBP_KD_Target") == "CTRL")
                        num_yogi = len(data)
                    elif dataset == "ayan":
                        data = data.filter(pl.col("RBP_KD") == "NONE")
                        num_ayan = len(data)
                    
                    data = data.select(pl.col("^.*_binding$"))
                    print(data.shape)

                binding_amount = data.sum()

                if not column_normalized: 
                    binding_amount = (binding_amount/len(data))*100

                binding_amount = binding_amount.to_pandas().T

                for idx, value in binding_amount.iterrows():
                    key, subkey = idx.split('_')[0:2]

                    if key not in heatmap_data[dataset]:
                        heatmap_data[dataset][key] = {}

                    heatmap_data[dataset][key][subkey] = value[0]

                heatmap_data[dataset] = pd.DataFrame.from_dict(heatmap_data[dataset])

            # Print the difference in columns
            unique_to_yogi = sorted(set(heatmap_data["yogi"].columns) - set(heatmap_data["ayan"].columns))
            unique_to_ayan = sorted(set(heatmap_data["ayan"].columns) - set(heatmap_data["yogi"].columns))

            logger.info(f"Columns unique to Yogi in {cell_line}: {unique_to_yogi}")
            logger.info(f"Columns unique to Ayan in {cell_line}: {unique_to_ayan}")

            common_columns = sorted(set(heatmap_data["yogi"].columns).intersection(set(heatmap_data["ayan"].columns)))
            heatmap_data["yogi"] = heatmap_data["yogi"][common_columns]
            heatmap_data["ayan"] = heatmap_data["ayan"][common_columns]

            if column_normalized: 
                heatmap_data["yogi"] = heatmap_data["yogi"].div(heatmap_data["yogi"].sum(axis=0), axis=1) * 100
                heatmap_data["ayan"] = heatmap_data["ayan"].div(heatmap_data["ayan"].sum(axis=0), axis=1) * 100

            vmin = min(heatmap_data["yogi"].min().min(), heatmap_data["ayan"].min().min())
            vmax = max(heatmap_data["yogi"].max().max(), heatmap_data["ayan"].max().max())

            if transcriptome_only:
                title_str = "Unique 3-Exon Combinations Only"
                num_yogi = len(self.transcriptome_only['yogi'][cell_line])
                num_ayan = len(self.transcriptome_only['ayan'][cell_line])

            elif not transcriptome_only:
                title_str = "All CTRL Data"

            fig, ax = plt.subplots(2, 1, figsize=(40, 12), dpi=200)
            
            sns.heatmap(heatmap_data["yogi"], ax=ax[0], cmap="Reds", vmin=vmin, vmax=vmax, cbar_kws={'shrink': 1.1, 'aspect': 10, 'pad': 0.01})
            ax[0].set_title(f"Yogi ({title_str}: {num_yogi})", fontsize=30, pad=20)
            ax[0].tick_params(axis='both', which='major', labelsize=15)
            cbar = ax[0].collections[0].colorbar
            cbar.ax.tick_params(labelsize=15)

            sns.heatmap(heatmap_data["ayan"], ax=ax[1], cmap="Reds", vmin=vmin, vmax=vmax, cbar_kws={'shrink': 1.1, 'aspect': 10, 'pad': 0.01})
            ax[1].set_title(f"Ayan ({title_str}: {num_ayan})", fontsize=30, pad=20)
            ax[1].tick_params(axis='both', which='major', labelsize=15)
            cbar = ax[1].collections[0].colorbar
            cbar.ax.tick_params(labelsize=15)

            if column_normalized: 
                analysis_type = "Positional Preference (Column Normalized)"
            elif not column_normalized: 
                analysis_type = "% Total Binding"

            plt.suptitle(f"{cell_line}- {analysis_type} - Yogi vs Ayan ({title_str})\nNOTE: only showing RBPs in both datasets", fontsize=40, y=1.02, x=0.4)

            plt.tight_layout()
            plt.savefig(f"yogi_vs_ayan_{cell_line}_heatmap_column_normalized_{column_normalized}_transcriptome_only_{transcriptome_only}.png", bbox_inches='tight')
            plt.close()