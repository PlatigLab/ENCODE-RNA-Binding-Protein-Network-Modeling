import pickle, argparse, gc

import xgboost as xgb, polars as pl, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

XGBOOST_BEST_MODEL_HASHES = {
    "HepG2": "fdf52464ba1145bed424d92827c559d851550a0117d96a630474c08e558ac4fd",
    "K562": "8f29591764f9e0de183047a4da90dca42b0f2847784de62970a3f20930cbe0db"
}
MODEL_DIR = "../outputs/pickled_models/XGBRegressor"
PREDICTIONS_DIR = "../outputs/predictions/XGBRegressor"


BACKGROUND_TYPES = {
    "non-sampling_based_backgrounds": [
        "ubp_less_than_.5",
        "ubp_less_than_.8", 
        "top_20_percent_dense_UBPs", 
        "lowest_20_percent_dense_UBPs", 
        "all_data", 
    ],
    "sampling_based_backgrounds": [
        "sample_100_per_PSI_bin",
        "sample_200_per_PSI_bin",
        "sample_500_per_PSI_bin", 
    ]
}

VALID_BACKGROUND_DATA_TYPES = [
    bg_type
    for group in BACKGROUND_TYPES.values()
    for bg_type in group
]

PLOT_COMMAND_OPTIONS = [
    "plot_average_predictions_for_non-sampling_based_backgrounds",
    "plot_average_predictions_for_sampling_based_backgrounds",
]

SEEDS = list(range(0, 1000, 10))


@dataclass
class ShapBackgroundTester:

    cell_line: str = None
    shap_background_data_type: str = None
    plot_command: str = None
    seed: int = None

    def __post_init__(self):


        if self.shap_background_data_type: 
            pass
        elif self.plot_command: 
            
            if self.plot_command.startswith("plot_average_predictions_for_"):
                self.plot_average_predictions_for_background_type(
                    mode = self.plot_command.replace("plot_average_predictions_for_", "")  
                )


    def load_model(self): 
        pass


    def get_binding_columns(self, df): 
        if type(df) == pl.DataFrame:
            binding_cols = [col for col in df.columns if col.endswith("_binding")]
        elif type(df) ==pl.LazyFrame:
            binding_cols = [col for col in df.collect_schema().names() if col.endswith("_binding")]
        
        return binding_cols

    
    def load_data_lazy(self, unique_binding=None):

        assert unique_binding in [True, False], "unique_binding must be True or False"
        
        lazyframes = {}
        for cell_line in XGBOOST_BEST_MODEL_HASHES.keys():
            lf = pl.scan_ipc(
                f"{PREDICTIONS_DIR}/{XGBOOST_BEST_MODEL_HASHES[cell_line]}.feather", 
            )

            binding_cols = self.get_binding_columns(lf)
            # Cast all columns ending with "_binding" to uint8
            lf = lf.with_columns([
                pl.col(col).cast(pl.UInt8) for col in binding_cols
            ])

            if unique_binding: 
                lf = self.convert_data_to_unique_binding(lf)
            
            lazyframes[cell_line] = lf
        
        return lazyframes
    

    def convert_data_to_unique_binding(self, df): 

        binding_cols = self.get_binding_columns(df)
        return df.unique(subset=binding_cols, keep="first", maintain_order=True)
    

    def retrieve_non_sampling_background_lazily(self, only_binding=None):
        assert self.shap_background_data_type in VALID_BACKGROUND_DATA_TYPES, f"shap_background_data_type must be one of {VALID_BACKGROUND_DATA_TYPES}"
        assert only_binding in [True, False], "only_binding must be True or False"

        lazyframes = self.load_data_lazy(unique_binding=False)
        lf = lazyframes[self.cell_line]

        if self.shap_background_data_type != "all_data":
            # Partition to Train and Validate
            lf = lf.filter(pl.col("Partition").is_in(["Train", "Validate"]))
            lf = self.convert_data_to_unique_binding(lf)


        if self.shap_background_data_type.startswith("ubp_less_than_"):
            threshold = float(self.shap_background_data_type.split("_")[-1])
            lf = lf.filter(pl.col("Predictions") < threshold)
            
        elif self.shap_background_data_type == "top_20_percent_dense_UBPs":
            binding_cols = self.get_binding_columns(lf)

            lf = lf.with_columns([
                pl.sum_horizontal([pl.col(col) for col in binding_cols]).alias("binding_sum")
            ])
            
            quantile = lf.select(pl.col("binding_sum").quantile(0.8)).collect()[0, 0]
            
            lf = lf.filter(pl.col("binding_sum") >= quantile)
            lf = lf.drop("binding_sum")
            
        elif self.shap_background_data_type == "lowest_20_percent_dense_UBPs":
            binding_cols = self.get_binding_columns(lf)

            lf = lf.with_columns([
                pl.sum_horizontal([pl.col(col) for col in binding_cols]).alias("binding_sum")
            ])
            
            quantile = lf.select(pl.col("binding_sum").quantile(0.2)).collect()[0, 0]
            
            lf = lf.filter(pl.col("binding_sum") <= quantile)
            lf = lf.drop("binding_sum")

        elif self.shap_background_data_type == "all_data":
            pass
        
        lf = lf.sort("index")
        
        if only_binding: 
            binding_cols = self.get_binding_columns(lf)
            lf = lf.select(binding_cols)
        
        return lf
        

    def sample_binding_patterns_by_PSI_bin(self, df):
        assert type(df) == pl.DataFrame, "df must be a polars DataFrame"
        assert self.shap_background_data_type.startswith("sample_"), "shap_background_data_type must start with 'sample_'"
        assert self.seed is not None, "seed must be set for sampling"
        
        num_per_bin = int(self.shap_background_data_type.split("_")[1])
        
        # Collect predictions and indices
        subset_df = df.select(["Predictions", "index"])
        rng = np.random.default_rng(self.seed)
        
        sampled_indices = []
        # Go through ranges [x, x+0.1) from 0.0 to 0.9
        for start in np.arange(0.0, 1.0, 0.1):
            
            end = start + 0.1
            # Filter for predictions in [start, end)
            bin_df = subset_df.filter(
                (subset_df["Predictions"] >= start) & (subset_df["Predictions"] < end)
            )
            bin_indices = bin_df["index"].to_numpy()
            
            assert len(bin_indices) > num_per_bin, f"Not enough samples in bin [{start}, {end}) to sample {num_per_bin} without replacement."

            sampled = rng.choice(bin_indices, size=num_per_bin, replace=False)
            sampled_indices.extend(sampled.tolist())

        return df.filter(pl.col("index").is_in(sampled_indices)).sort("index")


    def plot_average_predictions_for_background_type(self, mode = None): 
        assert mode in ["sampling_based_backgrounds", "non-sampling_based_backgrounds"], "mode must be either 'sampling_based_backgrounds' or 'non-sampling_based_backgrounds'"
        
        OUTPUT_DICT = {
            "sampling_based_backgrounds": {
                "data": "../outputs/background_data_experiments/predictions_across_backgrounds/sampling_based_backgrounds.tsv",
                "plot": "../outputs/background_data_experiments/predictions_across_backgrounds/sampling_based_backgrounds.png"
            },
            "non-sampling_based_backgrounds": {
                "data": "../outputs/background_data_experiments/predictions_across_backgrounds/non-sampling_based_backgrounds.tsv",
                "plot": "../outputs/background_data_experiments/predictions_across_backgrounds/non-sampling_based_backgrounds.png"
            }
        }

        DATA_FILE = OUTPUT_DICT[mode]["data"]

        if Path(DATA_FILE).exists():
            final_df = pd.read_csv(DATA_FILE, sep="\t")
            
            plt.figure(figsize=(18, 6), dpi=100)
            
            if mode == "non-sampling_based_backgrounds":
                
                ax = sns.barplot(
                    data=final_df,
                    x="cell_line",
                    y="mean_prediction",
                    hue="background_type",
                    palette="viridis",
                    edgecolor="black",
                    linewidth=1.5,
                    dodge=True
                )
            
            elif mode == "sampling_based_backgrounds":
                
                ax = sns.violinplot(
                    data=final_df,
                    x="cell_line",
                    y="mean_prediction",
                    hue="background_type",
                    palette="viridis",
                    inner=None,
                    linewidth=1.5,
                    dodge=True, 
                    density_norm="width"
                )

                sns.boxplot(
                    data=final_df,
                    x="cell_line",
                    y="mean_prediction",
                    hue="background_type",
                    palette="viridis",
                    showcaps=True,
                    boxprops={'facecolor':'none', "zorder": 10},
                    showfliers=False,
                    whiskerprops={'linewidth':2},
                    linewidth=1.5,
                    dodge=True,
                    ax=ax
                )
                
            
            plt.title("Average Predictions Across Background Types for Each Cell Line")
            plt.xlabel("Cell Line")
            plt.ylabel("Average Prediction")

            # Move legend outside and increase font size, set legend title to previous x-label
            handles, labels = ax.get_legend_handles_labels()
            # Remove duplicate labels
            unique = dict()
            for h, l in zip(handles, labels):
                if l not in unique:
                    unique[l] = h
            plt.legend(
                list(unique.values()),
                list(unique.keys()),
                title="Background Data Type",
                bbox_to_anchor=(1.02, 0.7),
                loc='upper left',
                borderaxespad=0.,
                fontsize=14,
                title_fontsize=16
            )

            plt.tight_layout(rect=[0, 0, 0.9, 1])
            plt.savefig(
                OUTPUT_DICT[mode]["plot"], 
                dpi=300, 
                bbox_inches='tight'
            )
            plt.close()
            
        else: 
            backgrounds = BACKGROUND_TYPES[mode]

            results = []
            for cell_line in XGBOOST_BEST_MODEL_HASHES.keys():
                self.cell_line = cell_line
                
                if mode == "sampling_based_backgrounds":
                    df = self.load_data_lazy(unique_binding=True)[cell_line].collect()
                    assert df.height > 0, f"No data found for cell line {cell_line} with background {backgrounds}"

                for background_type in tqdm(backgrounds, desc=f"Processing {cell_line}"): 
                    self.shap_background_data_type = background_type                        

                    if mode == "non-sampling_based_backgrounds": 
                        self.seed = None
                        lf = self.retrieve_non_sampling_background_lazily(only_binding=False)

                        predictions = lf.select("Predictions").collect()
                        assert predictions.height > 0, f"No data found for cell line {cell_line} with background {background_type}"
                        
                        mean_prediction = predictions["Predictions"].mean()
                        num_rows = predictions.height
                        results.append({
                            "cell_line": cell_line,
                            "background_type": background_type,
                            "mean_prediction": mean_prediction,
                            "num_rows": num_rows,
                        })
                        
                    elif mode == "sampling_based_backgrounds":
                        
                        for seed in tqdm(SEEDS, desc=f"Sampling seeds for {cell_line} with background {background_type}"):
                            self.seed = seed
                            sampled_df = self.sample_binding_patterns_by_PSI_bin(df)
                            
                            predictions = sampled_df["Predictions"]
                            assert len(predictions) > 0, f"No data found for cell line {cell_line} with background {background_type} and seed {seed}"
                            
                            mean_prediction = predictions.mean()
                            num_rows = len(predictions)
                            
                            results.append({
                                "cell_line": cell_line,
                                "background_type": background_type,
                                "mean_prediction": mean_prediction,
                                "num_rows": num_rows,
                                "seed": seed,
                            })

            final_df = pd.DataFrame(results)
            final_df.to_csv(DATA_FILE, sep="\t", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Vary background data for interventional probability SHAP.")
    parser.add_argument("--cell_line", type=str, choices=XGBOOST_BEST_MODEL_HASHES.keys(), help="Cell line to use.")
    parser.add_argument("--shap_background_data_type", type=str, choices=VALID_BACKGROUND_DATA_TYPES, help="Type of background data for SHAP.")
    parser.add_argument("--plot_command", type=str, choices=PLOT_COMMAND_OPTIONS, help="Plot command to execute.")
    parser.add_argument("--seed", type=int, help="Random seed.")

    args = parser.parse_args()

    if args.plot_command: 
        ShapBackgroundTester(
            plot_command=args.plot_command,
        )