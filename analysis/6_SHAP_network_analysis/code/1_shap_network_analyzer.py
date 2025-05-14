import glob, os, json, gc, pickle, gzip, tempfile, shutil, tqdm

import pandas as pd, polars as pl, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import scipy.cluster.hierarchy as sch

from dataclasses import dataclass
from IPython.display import display, Video
from loguru import logger
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr, spearmanr


@dataclass
class ShapNetworkInvestigator:
    PARAMS_DIR = "../../3_choose_dataset_and_model_parameters/output/model_reproduction/model_parameters/"
    MODEL_PICKLE_DIR = "../../4_run_final_models_and_SHAP/outputs/pickled_models/"
    CORRECTED_HAS_RBP_KD_DIR = "../../1_create_RBP_ML_input/4_create_num_peaks_ML_input/corrected_has_RBP_KD_output/"

    SHAP_MODEL_PICKLE_DIR = "../../4_run_final_models_and_SHAP/outputs/SHAP/regular/normal/explainer_objects/"
    SHAP_DIR = "../../4_run_final_models_and_SHAP/outputs/SHAP/regular/normal/shap_values/"
    SHAP_TYPE = "regular-observational"

    FDR_THRESHOLD = 0.1
    DPSI_THRESHOLD = 0.1

    CACHE_INFO = {
            "hash_metadata": "../outputs/hash_metadata/hash_metadata.tsv",
            # "SHAP_mp4": {
            #     "K562": "../outputs/local_SHAP_distribution_video/K562_local_SHAP_distribution.mp4",
            #     "HepG2": "../outputs/local_SHAP_distribution_video/HepG2_local_SHAP_distribution.mp4",
            # },
            "SHAP_CV": "../outputs/SHAP_cv/local_SHAP_cv.pkl.gz",
            'SHAP_cv_mp4': {
                "K562": "../outputs/video_plots/SHAP_cv/SHAP_cv_K562.mp4",
                "HepG2": "../outputs/video_plots/SHAP_cv/SHAP_cv_HepG2.mp4",
            },
            "SHAP_std": "../outputs/SHAP_std/local_SHAP_std.pkl.gz",
            'SHAP_std_mp4': {
                "K562": "../outputs/video_plots/SHAP_std/SHAP_std_K562.mp4",
                "HepG2": "../outputs/video_plots/SHAP_std/SHAP_std_HepG2.mp4",
            },
            "global_SHAP": {
                "5_dfs": "../outputs/global_SHAP/5_dfs_global_SHAP.pkl", 
                "5_dfs_average": "../outputs/global_SHAP/5_dfs_average_global_SHAP.pkl",
            },
            "local_SHAP_mean_vs_variance": {
                "K562": "../outputs/local_SHAP_mean_vs_variance/K562_local_SHAP_mean_vs_variance.png",
                "HepG2": "../outputs/local_SHAP_mean_vs_variance/HepG2_local_SHAP_mean_vs_variance.png"
            },
            "feature_metric_summary_table": "../outputs/feature_metric_summary_table/feature_metric_summary_table.tsv",
        }

    def __post_init__(self):

        self.make_model_metadata_from_hash()
        self.cell_lines = sorted(self.hash_metadata['cell_line'].unique().tolist())
    

    def make_model_metadata_from_hash(self):
        HASH_FILE = self.CACHE_INFO["hash_metadata"]

        if os.path.exists(HASH_FILE):
            hash_metadata = pd.read_csv(HASH_FILE, sep="\t")
            logger.success(f"FROM CACHE: loaded hash metadata file")

        else: 
            # Get all JSON files in the directory
            json_files = glob.glob(f"{self.PARAMS_DIR}/*.json")
            assert len(json_files) == 12, "10 XGBoost and 2 ElasticNet JSON files should be present"

            # List to hold dictionaries
            data = []

            # Process each JSON file
            for file_path in json_files:
                with open(file_path, 'r') as f:
                    file_data = json.load(f)
                # Add the 'hash' key
                file_data['hash'] = file_path.split('/')[-1].split('.')[0]
                data.append(file_data)

            # Add SHAP expected value for each XGBoost-related hash
            for file_data in data:
                if file_data['name'] == "XGBRegressor":
                    model_pickle_file = f"{self.SHAP_MODEL_PICKLE_DIR}/{file_data['hash']}.pkl"
                    with open(model_pickle_file, 'rb') as f:
                        model = pickle.load(f)
                    file_data['shap_expected_value'] = model.expected_value

            hash_metadata = pd.DataFrame(data)
            hash_metadata = hash_metadata.set_index('hash').reset_index().sort_values(by=['name', 'cell_line'])
            # Save to CSV
            hash_metadata.to_csv(f"{HASH_FILE}", sep="\t", index=False)
            logger.success(f"Created hash metadata file")
            
        self.hash_metadata = hash_metadata

    
    def calculate_pointwise_SHAP_metric_per_cell_line(self, cell_line_shap=None, metric=None):
        
        assert None not in (cell_line_shap, metric), "Arguments 'cell_line_shap' and 'metric' cannot be None"
        logger.info(f"Calculating pointwise SHAP metric -- {metric}")

        # Ensure we have 5 dataframes
        assert len(cell_line_shap) == 5

        # Sort each DataFrame in cell_line_shap by the "index" column
        cell_line_shap = [df.sort("index") for df in cell_line_shap]
        # Ensure that the order of the indices matches exactly across all 5 DataFrames
        assert all(
            [(df["index"].to_list() == cell_line_shap[0]["index"].to_list()) for df in cell_line_shap]
        ), "Index order mismatch across SHAP DataFrames"
        cell_line_shap = [df.drop("index") for df in cell_line_shap]

        # ensure all dataframes have the same shape
        assert all(df.shape == cell_line_shap[0].shape for df in cell_line_shap), "SHAP DataFrames have inconsistent dimensions"
        # ensure all dataframes have the same columns and ordering
        assert all(df.columns == cell_line_shap[0].columns for df in cell_line_shap), "Column names are not consistent across cell_line_shap"

        # Convert to numpy tensors and stack
        tensors = np.stack([df.to_numpy() for df in cell_line_shap], axis=0)

        # Calculate the desired metric along the stacked axis
        if metric == 'std':
            result = np.std(tensors, axis=0)
        elif metric == 'mean':
            result = np.mean(tensors, axis=0)
        elif metric == 'median':
            result = np.median(tensors, axis=0)
        elif metric =='coefficient_of_variation':
            mean = np.mean(tensors, axis=0)
            std = np.std(tensors, axis=0)
            result = std / mean
        elif metric == 'variance':
            result = np.var(tensors, axis=0)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Convert the result back to a Polars DataFrame
        result_df = pl.DataFrame(result, schema=cell_line_shap[0].columns)
        # Assert that the result and result_df have the same number of columns and rows as the second DataFrame in cell_line_shap
        # using second dataframe just as another double check that the first and second dataframe have the same shape
        assert result.shape == cell_line_shap[1].shape, "Resulting metric array has inconsistent dimensions"
        assert result_df.shape == cell_line_shap[1].shape, "Resulting DataFrame has inconsistent dimensions"

        logger.success(f"Calculated pointwise SHAP {metric}")
        return result_df

    
    def retrieve_5_SHAP_tables_per_cell_line(self, cell_line):
        
        # Filter hash metadata for rows where 'name' contains 'xgboost'
        xgboost_metadata = self.hash_metadata[self.hash_metadata['name']=="XGBRegressor"]
        group = xgboost_metadata[xgboost_metadata['cell_line'] == cell_line]
        assert len(group) == 5, f"Expected 5 SHAP files for cell line {cell_line}, but found {len(group)}"

        # Group by cell_line
        logger.info(f"Loading SHAP files for cell line {cell_line}")

        shap_dfs = []
        for hash_value in group['hash']:
            feather_file = f"{self.SHAP_DIR}/{hash_value}.feather"

            # Read the feather file using polars
            df = pl.scan_ipc(feather_file)
            
            schema = df.collect_schema().names()
            # Subset to all columns in schema that end in "_shap" and the "index" column
            shap_columns = [col for col in schema if col.endswith("_shap")] + ["index"]

            df = df.select(shap_columns).sort('index').collect()
            
            logger.info(f"Loaded SHAP file for cell line {cell_line} with hash {hash_value} and shape {df.shape}")
            shap_dfs.append(df)

        return shap_dfs
    

    def get_RBP_position(self, name): 
        name = name.split("_")
        assert len(name)==3, logger.error(f"Expected 3 parts in the name {name}, but got {len(name)}")
        return name[0], int(name[1])
    

    def convert_RBP_position_to_2d_heatmap(self, table):

        assert len(table) == 1, "Table should have only one row"
        if not isinstance(table, pd.DataFrame):
            assert isinstance(table, pl.DataFrame), "Table must be either a pandas or polars DataFrame"
            table = table.to_pandas()

        # Extract RBP and position information from column names
        rbp_positions = [self.get_RBP_position(col) for col in table.columns]

        # Create a DataFrame with RBP and position as separate columns
        rbp_positions_df = pd.DataFrame(
            [(rbp, position, table[col].iloc[0]) for (rbp, position), col in zip(rbp_positions, table.columns)],
            columns=["RBP", "Position", "Value"]
        )

        # Pivot the DataFrame to create a 2D heatmap
        heatmap_df = rbp_positions_df.pivot(index="Position", columns="RBP", values="Value")
        # Sort the columns (RBPs) and rows (positions)
        heatmap_df = heatmap_df.sort_index(axis=0).sort_index(axis=1)

        return heatmap_df


    def delete_data(self, data_type=None):
        assert data_type is not None, "data_type cannot be None"

        if data_type == 'SHAP_cv': 
            assert hasattr(self, 'SHAP_cv'), "SHAP_cv attribute does not exist"
            del self.SHAP_cv
            logger.success("Deleted SHAP_cv attribute")

        elif data_type == 'SHAP_std':
            assert hasattr(self, 'SHAP_std'), "SHAP_std attribute does not exist"
            del self.SHAP_std
            logger.success("Deleted SHAP_std attribute")

        gc.collect()


    # def plot_local_SHAP_distribution_per_feature_as_mp4(self): 

    #     if all(os.path.exists(output_file) for cell_line in self.CACHE_INFO["SHAP_mp4"] for output_file in self.CACHE_INFO["SHAP_mp4"][cell_line]):
    #         output_files = [self.CACHE_INFO["SHAP_mp4"][cell_line] for cell_line in self.cell_lines]
    #         logger.success("FROM CACHE: local SHAP distribution mp4s already exist.")

    #         # Load the mp4s from the cache
    #         for output_file in output_files:    
    #             with open(output_file, 'rb') as f:
    #                 mp4 = f.read()
    #             # Display the mp4
    #             plt.imshow(mp4)
    #             plt.axis('off')
    #             plt.show()

        
    #     else: 
    #         for cell_line in self.cell_lines:

    #             logger.info(f"Local SHAP distribution mp4 for {cell_line} does not exist. Calculating...")

    #             #TODO fix this at the end
    #             # Retrieve the 5 SHAP DataFrames for the cell line
    #             # shap_dfs = self.retrieve_5_SHAP_tables_per_cell_line(cell_line)
    #             # self.shap_dfs = shap_dfs

    #             shap_dfs = self.shap_dfs

    #             # Create a mp4 of the local SHAP distribution
    #             for feature in sorted(shap_dfs[0].columns):
    #                 fig, axes = plt.subplots(3, 2, figsize=(15, 10), dpi=300, sharex=True, sharey=True)

    #                 # Plot histograms for each SHAP table
    #                 for i, df in enumerate(shap_dfs):
    #                     ax = axes[i // 2, i % 2]
    #                     sns.histplot(df[feature].to_numpy(), bins=50, stat='percent', color='deepskyblue', edgecolor='black', alpha=0.7, ax=ax)
    #                     ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    #                     ax.set_title(f"Model {i + 1}")
    #                     ax.set_xlabel('')
    #                     ax.set_ylabel('')

    #                 # # Add a table in the 6th subplot
    #                 # ax = axes[2, 1]
    #                 # ax.axis('off')
    #                 # zero_percentages = [
    #                 #     (df[feature] == 0).sum() / len(df[feature]) * 100 for df in shap_dfs
    #                 # ]
    #                 # table_data = [[f"Table {i + 1}", f"{zero_percentage:.2f}%"] for i, zero_percentage in enumerate(zero_percentages)]
    #                 # ax.table(cellText=table_data, colLabels=["Table", "Zero %"], loc='center', cellLoc='center')

    #                 fig.suptitle(f"{cell_line} - {feature}", fontsize=30, y=0.98)
    #                 fig.supxlabel("Local SHAP Value", fontsize=20)
    #                 fig.supylabel("% of Values in Bin", fontsize=20)


    #                 plt.tight_layout(rect=[0, 0, 1, 0.95])
    #                 # plt.savefig(output_file.replace(".mp4", f"_{feature}.png"))
    #                 plt.show()
    #                 plt.close()
                
    #             logger.success(f"Saved local SHAP distribution mp4 for {cell_line} to {output_file}")


    def create_plot_movie_from_features(self, data, output_file): 

        if os.path.exists(output_file):
            logger.success(f"FROM CACHE: {output_file} already exists. Rendering...")
            # Display the video
            return display(Video(filename=output_file))

        else: 
            logger.info(f"{output_file} does not exist. Creating video...")

            # Create a temporary directory to store histogram images
            temp_dir = tempfile.mkdtemp()
            # Generate histogram plots for each column
            for i, column in enumerate(tqdm.tqdm(data.columns, desc="Generating histograms")):
                plt.figure(figsize=(8, 4))
                sns.histplot(data[column].to_numpy(), bins=50, stat="percent", color="deepskyblue", edgecolor="black", alpha=0.7)
                
                plt.axvline(x=0, color="red", linestyle="--", linewidth=2)
                plt.title(f"{column}", fontsize=16)
                plt.xlabel("Value", fontsize=14)
                plt.ylabel("Percentage", fontsize=14)
                plt.tight_layout()
                
                # Save the plot as an image
                image_path = os.path.join(temp_dir, f"{column}.png")
                plt.savefig(image_path, dpi=150)
                plt.close()

            # Create a video from the saved images
            image_files = sorted(glob.glob(os.path.join(temp_dir, "*.png")))
            clip = ImageSequenceClip(image_files, fps=2)
            clip.write_videofile(output_file, codec="libx264", fps=2)

            logger.success(f"Saved video to {output_file}")

            # Clean up the temporary directory
            shutil.rmtree(temp_dir)


    def calculate_global_SHAP(self, mode=None): 
        assert mode in ['5_dfs', '5_dfs_average'], "mode should be either '5_dfs' or '5_dfs_average'"

        output_file = self.CACHE_INFO["global_SHAP"][mode]
        # Check if the output file exists
        if os.path.exists(output_file):
            with open(output_file, 'rb') as f:
                global_SHAP = pickle.load(f)
            logger.success(f"FROM CACHE: loaded global SHAP file for mode {mode}")
            return global_SHAP
        
        else:
            logger.info(f"Global SHAP file for mode {mode} does not exist. Calculating...")

            global_heatmaps = {}
            dfs_global_SHAP = {
                cell_line: self.retrieve_5_SHAP_tables_per_cell_line(cell_line) for cell_line in self.cell_lines
            }

            if mode == '5_dfs':
                for cell_line, dfs in dfs_global_SHAP.items():
                    # Drop the 'index' column and calculate the absolute value average of each column
                    averaged_df = [df.sort('index').drop('index').select(pl.all().abs().mean()) for df in dfs]
                    # Convert each averaged DataFrame to a 2D heatmap
                    heatmaps = [self.convert_RBP_position_to_2d_heatmap(df) for df in averaged_df]
                    # Store the heatmaps in the global dictionary
                    global_heatmaps[cell_line] = heatmaps

            elif mode == '5_dfs_average':
                for cell_line in self.cell_lines: 
                    average_df = self.calculate_pointwise_SHAP_metric_per_cell_line(
                        cell_line_shap=dfs_global_SHAP[cell_line], 
                        metric='mean'
                    )

                    average_df = average_df.select(pl.all().abs().mean())
                    heatmap = self.convert_RBP_position_to_2d_heatmap(average_df)
                    # Store the heatmap in the global dictionary
                    global_heatmaps[cell_line] = heatmap
            
            assert len(global_heatmaps.keys()) == len(self.cell_lines), "Not all cell lines have been processed"
            # Save the global heatmaps as a single pickle file
            with open(output_file, 'wb') as f:
                pickle.dump(global_heatmaps, f)

            logger.success(f"Saved global SHAP file for mode {mode} to {output_file}")


    def plot_global_SHAP(self, mode=None):
        assert mode in ['5_dfs', '5_dfs_average'], "mode should be either '5_dfs' or '5_dfs_average'"
        global_SHAP = self.calculate_global_SHAP(mode)

        if mode == '5_dfs':

            for cell_line, heatmaps in global_SHAP.items():
                fig, axes = plt.subplots(3, 2, figsize=(45, 15), dpi=300, sharex=True, sharey=True)
                fig.suptitle(f"{cell_line}: Top 5 Models Global SHAP", fontsize=40)
                # Flatten axes for easier iteration
                axes = axes.flatten()
                # Determine global min and max values for consistent color scaling
                vmin = min(heatmap.min().min() for heatmap in heatmaps)
                vmax = max(heatmap.max().max() for heatmap in heatmaps)

                for i, heatmap in enumerate(heatmaps):
                    sns.heatmap(
                        heatmap,
                        ax=axes[i],
                        cmap="Blues",
                        cbar=False,  # Disable individual colorbars
                        vmin=vmin,
                        vmax=vmax
                    )
                    axes[i].set_title(f"Model {i + 1}")
                    axes[i].set_xlabel("RBP")
                    axes[i].set_ylabel("Position")

                # Hide any unused subplots
                for j in range(len(heatmaps), len(axes)):
                    axes[j].axis("off")

                # Add a single colorbar for the entire figure
                cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])  # Position for the colorbar
                sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=vmin, vmax=vmax))
                sm.set_array([])
                cbar = fig.colorbar(sm, cax=cbar_ax)
                cbar.ax.tick_params(labelsize=25)  # Set the font size for the colorbar labels

                plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # Adjust layout to make space for the colorbar
                plt.show()
                plt.close()

        elif mode == '5_dfs_average':

            # Calculate the combined range of all heatmaps to define consistent bins
            all_values = np.concatenate([heatmap.to_numpy().flatten() for heatmap in global_SHAP.values()])
            bins = np.linspace(all_values.min(), all_values.max(), 21)  # Define 20 equal-width bins

            # Create a figure with 2 columns: left for histograms, right for boxplots
            fig, axes = plt.subplots(len(global_SHAP), 2, figsize=(11, 7), dpi=200, sharex=True, sharey=False)

            for row_idx, (cell_line, heatmap) in enumerate(global_SHAP.items()):
                # Flatten the heatmap values into a single array
                global_shap_values = heatmap.to_numpy().flatten()
                num_points = len(global_shap_values)

                # Left subplot: histogram
                sns.histplot(
                    global_shap_values,
                    bins=bins,
                    stat="percent",
                    color="deepskyblue",
                    edgecolor="black",
                    alpha=0.7,
                    ax=axes[row_idx, 0]
                )
                axes[row_idx, 0].set_title(f"{cell_line} (n={num_points})", fontsize=12)
                axes[row_idx, 0].set_xlabel("")
                axes[row_idx, 0].set_ylabel("")
                axes[row_idx, 0].tick_params(axis="both", labelsize=12)

                # Right subplot: boxplot with dots
                sns.boxplot(
                    data=global_shap_values,
                    orient="h",
                    color="deepskyblue",
                    ax=axes[row_idx, 1],
                    width=0.5,
                    showmeans=True,
                    meanline=True,
                    meanprops={"color": "red", "linewidth": 1.5}
                )
                sns.stripplot(
                    data=global_shap_values,
                    orient="h",
                    color="black",
                    size=5,
                    alpha=0.1,
                    ax=axes[row_idx, 1]
                )
                axes[row_idx, 1].set_title(f"{cell_line} (n={num_points})", fontsize=12)
                axes[row_idx, 1].set_xlabel("")
                axes[row_idx, 1].set_ylabel("")
                axes[row_idx, 1].tick_params(axis="both", labelsize=12)

            plt.suptitle("Global SHAP Values per Cell Line from Averaging 5 Models' Local SHAP", fontsize=20, y=0.98)
            fig.supxlabel("Global SHAP Value", fontsize=16)
            fig.supylabel("Percentage", fontsize=16)
            plt.tight_layout()
            plt.show()

            for iteration, vmin_threshold in enumerate([None, 0.05]):
                fig, axes = plt.subplots(2, 1, figsize=(35, 13), dpi=200, sharey=True)

                for ax, (cell_line, heatmap) in zip(axes, global_SHAP.items()):
                    # Perform hierarchical clustering on the columns
                    linkage = sch.linkage(heatmap.T, method="ward")
                    dendrogram = sch.dendrogram(linkage, no_plot=True)
                    ordered_columns = [heatmap.columns[i] for i in dendrogram["leaves"]]

                    # Reorder the heatmap columns based on the clustering
                    ordered_heatmap = heatmap[ordered_columns]

                    sns.heatmap(
                        ordered_heatmap,
                        ax=ax,
                        cmap="Blues",
                        cbar=True,
                        linewidths=0.01,  # Add black border around each cell
                        linecolor="gray",
                        cbar_kws={"shrink": 1, "aspect": 20, "pad": 0.02},  # Adjust colorbar position and size
                        vmin=vmin_threshold  # Apply minimum threshold for the second iteration
                    )
                    cbar = ax.collections[0].colorbar
                    cbar.ax.tick_params(labelsize=20)  # Make colorbar tick labels larger
                    ax.set_title(f"{cell_line}", fontsize=30)
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.tick_params(axis='y', labelsize=25)  # Make y-axis tick labels larger

                # Add a caption for the second iteration
                caption = ""
                if vmin_threshold is not None:
                    caption = f"\nCAVEAT: colorbar set to minimum of {vmin_threshold}"

                fig.suptitle(
                    f"Global SHAP w/ Ward Hierarchical Clustering Order\n"
                    f"(NOTE: after averaging all local SHAP values across 5 models per cell line){caption}",
                    fontsize=40, y=1.01, x=0.45
                )
                fig.supxlabel("RBP", fontsize=30)
                fig.supylabel("Position", fontsize=30, x=-0.01)
                plt.tight_layout()
                plt.show()

            # Convert global_SHAP data into pandas DataFrames
            hepg2_df = global_SHAP["HepG2"].copy()
            k562_df = global_SHAP["K562"].copy()

            # Extract intersecting RBPs and positions
            intersecting_rbps = hepg2_df.columns.intersection(k562_df.columns)
            intersecting_positions = hepg2_df.index.intersection(k562_df.index)

            # Subset the DataFrames to intersecting RBPs and positions using a for loop
            hepg2_values = []
            k562_values = []
            features = []

            for position in intersecting_positions:
                for rbp in intersecting_rbps:
                    hepg2_values.append(hepg2_df.at[position, rbp])
                    k562_values.append(k562_df.at[position, rbp])
                    features.append(f"{rbp}_{position}")

            # Create a combined DataFrame for plotting
            combined_df = pd.DataFrame({
                "HepG2": hepg2_values,
                "K562": k562_values,
                "Feature": features
            })

            # Identify the top 5 features with the highest global SHAP values in HepG2 and K562
            top_hepg2_features = combined_df.nlargest(5, "HepG2")
            top_k562_features = combined_df.nlargest(5, "K562")

            # Combine the top features for annotation
            top_features = pd.concat([top_hepg2_features, top_k562_features]).drop_duplicates()

            # Calculate correlations
            pearson_corr, _ = pearsonr(hepg2_values, k562_values)
            spearman_corr, _ = spearmanr(hepg2_values, k562_values)

            # Create scatterplot
            plt.figure(figsize=(6,4), dpi=200)
            sns.scatterplot(
                x=hepg2_values,
                y=k562_values,
                alpha=0.7,
                edgecolor="black",
                color="deepskyblue",
                s=20
            )

            # Annotate top features
            for _, row in top_features.iterrows():
                plt.text(
                    row["HepG2"] - 0.01, 
                    row["K562"] + 0.005, 
                    row["Feature"], 
                    fontsize=6, 
                    color="black", 
                    alpha=0.8
                )

            # Add y=x line
            plt.plot(
                [min(hepg2_values), max(hepg2_values)],
                [min(hepg2_values), max(hepg2_values)],
                color="red",
                linestyle="--",
                linewidth=1,
                label="y=x"
            )

            # Add annotations
            plt.title("Global SHAP: HepG2 vs K562", fontsize=14)
            plt.xlabel("HepG2 Global SHAP", fontsize=12)
            plt.ylabel("K562 Global SHAP", fontsize=12)
            plt.text(
                0.4, 0.95,
                f"Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}\nPoints: {len(hepg2_values)}",
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment='top'
            )
            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.show()

    
    def plot_global_SHAP_mean_vs_variance(self, mode=None):
        assert mode in ['5_dfs', '5_dfs_average'], "mode should be either '5_dfs' or '5_dfs_average'"
        global_SHAP = self.calculate_global_SHAP(mode)

        if mode == '5_dfs':

            return_results = {}
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300, sharex=True, sharey=True)

            for ax, (cell_line, heatmaps) in zip(axes, global_SHAP.items()):
                # Assert that all heatmaps have the same shape and ordering
                assert all(heatmap.shape == heatmaps[0].shape for heatmap in heatmaps), "Heatmaps have inconsistent dimensions"
                assert all(heatmap.columns.tolist() == heatmaps[0].columns.tolist() for heatmap in heatmaps), "Column ordering mismatch in heatmaps"
                assert all(heatmap.index.tolist() == heatmaps[0].index.tolist() for heatmap in heatmaps), "Row ordering mismatch in heatmaps"

                # Stack all 5 heatmaps into a 3D numpy array
                stacked_heatmaps = np.stack([heatmap.to_numpy() for heatmap in heatmaps], axis=0)

                # Calculate the mean and variance for each RBP-Position combination
                mean_values = np.mean(stacked_heatmaps, axis=0)
                variance_values = np.var(stacked_heatmaps, axis=0)

                # Convert mean_values back to a DataFrame with row and column indices
                mean_values = pd.DataFrame(mean_values, index=heatmaps[0].index, columns=heatmaps[0].columns)
                variance_values = pd.DataFrame(variance_values, index=heatmaps[0].index, columns=heatmaps[0].columns)

                # Create a new table with RBP-Position combinations, mean, and variance
                rbp_positions = []
                mean_flat = []
                variance_flat = []

                for position in mean_values.index:
                    for rbp in mean_values.columns:
                        rbp_positions.append(f"{rbp}_{position}")
                        mean_flat.append(mean_values.at[position, rbp])
                        variance_flat.append(variance_values.at[position, rbp])

                result_table = pd.DataFrame({
                    "RBP_Position": rbp_positions,
                    "Mean": mean_flat,
                    "Variance": variance_flat
                })
                return_results[cell_line] = result_table.sort_values(by="Variance", ascending=False)

                # Plot mean vs variance on a scatterplot
                scatter = sns.scatterplot(data=result_table, x="Mean", y="Variance", alpha=0.7, edgecolor="black", color="deepskyblue", ax=ax)
                ax.set_title(f"{cell_line}", fontsize=14)
                ax.set_xlabel('')
                ax.set_ylabel('')

                # Format the y-axis tick labels to include more explicit zeros
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}"))

            plt.suptitle("Mean vs Variance of Global SHAP Values\nper Feature Across 5 Models", fontsize=16)
            fig.supxlabel("Mean of Global SHAP", fontsize=14)
            fig.supylabel("Variance of Global SHAP", fontsize=14)

            plt.tight_layout()
            plt.show()
            plt.close()

            return return_results


    def calculate_SHAP_CV(self):

        output_file = self.CACHE_INFO["SHAP_CV"]
        # Check if the output file exists
        if os.path.exists(output_file):
            with gzip.open(output_file, 'rb') as f:
                SHAP_cv = pickle.load(f)
            
            # Convert each DataFrame in SHAP_cv from pandas to polars
            self.SHAP_cv = {cell_line: pl.from_pandas(df) for cell_line, df in SHAP_cv.items()}
            logger.success(f"FROM CACHE: loaded SHAP CV file")   

        else:
            logger.info(f"SHAP CV file does not exist. Calculating...")

            # Initialize an empty dictionary to store SHAP CV results
            SHAP_CV = {}
            # Calculate the coefficient of variation for each cell line
            for cell_line in self.cell_lines:
                logger.info(f"Calculating SHAP CV for cell line {cell_line}")

                # Retrieve the 5 SHAP DataFrames for the cell line
                shap_dfs = self.retrieve_5_SHAP_tables_per_cell_line(cell_line)
                # Calculate the coefficient of variation using the pointwise metric function
                cv_df = self.calculate_pointwise_SHAP_metric_per_cell_line(
                    cell_line_shap=shap_dfs, 
                    metric='coefficient_of_variation'
                )
                
                SHAP_CV[cell_line] = cv_df.to_pandas()

            # Save the SHAP CV as a single gzip-compressed pickle file
            with gzip.open(output_file, 'wb') as f:
                pickle.dump(SHAP_CV, f)

            logger.success(f"Saved SHAP CV file to {output_file}")

    
    def plot_SHAP_CV(self):
        
        if not hasattr(self, 'SHAP_cv'):
            self.calculate_SHAP_CV()

        for cell_line, cv_df in self.SHAP_cv.items():
            logger.info(f"Plotting SHAP CV for cell line {cell_line}")
            
            # Subset to features (columns) that are not all null values
            valid_features = [col for col in cv_df.columns if not cv_df[col].is_null().all()]
            
            filtered_data = cv_df.select(valid_features)
            assert sum(filtered_data.null_count().row(0)) == 0, "Filtered data contains null values"

            # Generate the movie for the filtered data
            output_file = self.CACHE_INFO["SHAP_cv_mp4"][cell_line]
            self.create_plot_movie_from_features(filtered_data, output_file)

        self.delete_data(data_type='SHAP_cv')


    def calculate_SHAP_std(self):
        output_file = self.CACHE_INFO["SHAP_std"]
        # Check if the output file exists
        if os.path.exists(output_file):
            with gzip.open(output_file, 'rb') as f:
                SHAP_std = pickle.load(f)
            
            # Convert each DataFrame in SHAP_std from pandas to polars
            self.SHAP_std = {cell_line: pl.from_pandas(df) for cell_line, df in SHAP_std.items()}
            logger.success(f"FROM CACHE: loaded SHAP std file")   

        else:
            logger.info(f"SHAP std file does not exist. Calculating...")

            # Initialize an empty dictionary to store SHAP std results
            SHAP_std = {}
            # Calculate the coefficient of variation for each cell line
            for cell_line in self.cell_lines:
                logger.info(f"Calculating SHAP std for cell line {cell_line}")

                # Retrieve the 5 SHAP DataFrames for the cell line
                shap_dfs = self.retrieve_5_SHAP_tables_per_cell_line(cell_line)
                # Calculate the coefficient of variation using the pointwise metric function
                std_df = self.calculate_pointwise_SHAP_metric_per_cell_line(
                    cell_line_shap=shap_dfs, 
                    metric='std'
                )
                
                SHAP_std[cell_line] = std_df.to_pandas()

            # Save the SHAP std as a single gzip-compressed pickle file
            with gzip.open(output_file, 'wb') as f:
                pickle.dump(SHAP_std, f)

            logger.success(f"Saved SHAP std file to {output_file}")


    def plot_SHAP_std(self):
        if not hasattr(self, 'SHAP_std'):
            self.calculate_SHAP_std()

        for cell_line, std_df in self.SHAP_std.items():
            logger.info(f"Plotting SHAP std for cell line {cell_line}")
            
            # Subset to features (columns) that are not all null values
            valid_features = [col for col in std_df.columns if not std_df[col].is_null().all()]
            
            filtered_data = std_df.select(valid_features)
            assert sum(filtered_data.null_count().row(0)) == 0, "Filtered data contains null values"

            # Generate the movie for the filtered data
            output_file = self.CACHE_INFO["SHAP_std_mp4"][cell_line]
            self.create_plot_movie_from_features(filtered_data, output_file)

        self.delete_data(data_type='SHAP_std')

    
    def plot_local_SHAP_mean_vs_variance(self): 
        # Check if the local SHAP mean vs variance plots exist for all cell lines
        if all(os.path.exists(self.CACHE_INFO["local_SHAP_mean_vs_variance"][cell_line]) for cell_line in self.cell_lines):
            logger.success("FROM CACHE: Local SHAP mean vs variance plots already exist.")
            for cell_line in self.cell_lines:
                output_file = self.CACHE_INFO["local_SHAP_mean_vs_variance"][cell_line]

                logger.success(f"FROM CACHE: Local SHAP mean vs variance plot for {cell_line} already exists.")
                
                img = plt.imread(output_file)
                plt.figure(figsize=(6,7), dpi=300)  # Adjust the figure size as needed
                plt.imshow(img)
                plt.axis('off')  # Hide axes for better visualization
                plt.show()

        else:
            logger.info("Both local SHAP mean vs variance plots do not exist. Proceeding to generate the necessary ones...")

            for cell_line in self.cell_lines:
                
                # adding this as each cell line plot takes forever to generate
                if os.path.exists(self.CACHE_INFO["local_SHAP_mean_vs_variance"][cell_line]): 
                    logger.success(f"FROM CACHE: Local SHAP mean vs variance plot for {cell_line} already exists. Generating other cell line...")
                    continue
                
                logger.info(f"Generating mean vs variance hexbin plot for cell line {cell_line}")

                # Retrieve the 5 SHAP DataFrames for the cell line
                shap_dfs = self.retrieve_5_SHAP_tables_per_cell_line(cell_line)

                # Calculate mean and variance using the pointwise SHAP metric function
                mean_df = self.calculate_pointwise_SHAP_metric_per_cell_line(
                    cell_line_shap=shap_dfs, 
                    metric='mean'
                )
                variance_df = self.calculate_pointwise_SHAP_metric_per_cell_line(
                    cell_line_shap=shap_dfs, 
                    metric='variance'
                )

                # Flatten the mean and variance DataFrames for plotting
                mean_values = mean_df.to_numpy().flatten()
                variance_values = variance_df.to_numpy().flatten()

                del mean_df, variance_df, shap_dfs
                gc.collect()

                # Assert that there are no null or missing values in either mean or variance
                assert not np.isnan(mean_values).any(), "Mean values contain NaN or missing values"
                assert not np.isnan(variance_values).any(), "Variance values contain NaN or missing values"

                plt.figure(figsize=(6,3), dpi=300)
                hb = plt.hexbin(
                    mean_values, 
                    variance_values, 
                    gridsize=50, 
                    cmap='viridis', 
                    norm=LogNorm(), 
                    edgecolors='black', 
                    linewidths=0.05
                )
                
                plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5)

                plt.colorbar(hb, label='Log Counts').ax.tick_params(labelsize=8)
                plt.title(f"{cell_line}: Local SHAP Mean vs Variance\nNOTE: mean and variance calculated for each point in 5-stack tensor", fontsize=8)
                plt.xlabel("Mean SHAP Value", fontsize=8)
                plt.ylabel("Variance SHAP Value", fontsize=8)

                # Save the plot to the corresponding output file
                output_file = self.CACHE_INFO["local_SHAP_mean_vs_variance"][cell_line]
                plt.tight_layout()
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.show()
                plt.close()

                logger.success(f"Saved mean vs variance hexbin plot for {cell_line} to {output_file}")

    
    def load_elasticnet_coefficients(self): 

        # Filter hash metadata for rows where 'name' contains 'ElasticNet'
        elasticnet_metadata = self.hash_metadata[self.hash_metadata['name'] == "ElasticNet"].sort_values(by=['cell_line', 'hash'])
        # Check if the number of rows is 2
        assert len(elasticnet_metadata) == 2, "Expected 2 ElasticNet models for each cell line"

        # Initialize a dictionary to store ElasticNet information
        elasticnet_info = {}
        for _, row in elasticnet_metadata.iterrows():
            cell_line = row['cell_line']
            hash_value = row['hash']
            subfolder = row['name']

            # Construct the path to the pickle file
            pickle_file = os.path.join(self.MODEL_PICKLE_DIR, subfolder, f"{hash_value}.pkl.gz")

            # Load the ElasticNet model
            with gzip.open(pickle_file, 'rb') as f:
                model = pickle.load(f)

            # Log the number of iterations
            logger.info(f"Cell Line: {cell_line}, Hash: {hash_value}, n_iter_: {model.n_iter_}")
            # Assert that the order and content of feature_names_in_ matches column_order_when_fitting
            assert list(model.feature_names_in_) == list(model.column_order_when_fitting), f"Feature names mismatch for cell line {cell_line}, hash {hash_value}"

            # Store the coefficients as a DataFrame and intercept in the dictionary
            elasticnet_info[cell_line] = {
                "coefficients": pd.DataFrame([model.coef_], columns=model.feature_names_in_),
                "intercept": model.intercept_
            }
        
        self.elasticnet_info = elasticnet_info
        logger.success("Loaded ElasticNet coefficients and intercepts")
        return self.elasticnet_info

    
    def plot_elasticnet_coefficients(self):

        # Convert ElasticNet coefficients to 2D heatmaps
        heatmap_HepG2 = self.convert_RBP_position_to_2d_heatmap(self.elasticnet_info["HepG2"]["coefficients"])
        heatmap_K562 = self.convert_RBP_position_to_2d_heatmap(self.elasticnet_info["K562"]["coefficients"])

        # Create a figure with 2 columns: left for histograms, right for boxplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=200, sharex=True, sharey="col")

        for row_idx, (cell_line, heatmap) in enumerate({"HepG2": heatmap_HepG2, "K562": heatmap_K562}.items()):
            # Flatten the heatmap values into a single array
            elasticnet_values = heatmap.to_numpy().flatten()
            num_points = len(elasticnet_values)

            # Left subplot: histogram
            sns.histplot(
                elasticnet_values,
                bins=50,
                stat="percent",
                color="deepskyblue",
                edgecolor="black",
                alpha=0.7,
                ax=axes[row_idx, 0]
            )
            axes[row_idx, 0].set_title(f"{cell_line} (n={num_points})", fontsize=12)
            axes[row_idx, 0].set_xlabel("")
            axes[row_idx, 0].set_ylabel("")
            axes[row_idx, 0].tick_params(axis="both", labelsize=10)

            # Right subplot: boxplot with stripplot
            sns.boxplot(
                data=elasticnet_values,
                orient="h",
                color="deepskyblue",
                ax=axes[row_idx, 1],
                width=0.5,
                showmeans=True,
                meanline=True,
                meanprops={"color": "red", "linewidth": 1.5}
            )
            sns.stripplot(
                data=elasticnet_values,
                orient="h",
                color="black",
                size=5,
                alpha=0.1,
                ax=axes[row_idx, 1]
            )
            axes[row_idx, 1].set_title(f"{cell_line} (n={num_points})", fontsize=12)
            axes[row_idx, 1].set_xlabel("")
            axes[row_idx, 1].set_ylabel("")
            axes[row_idx, 1].tick_params(axis="both", labelsize=10)

        plt.suptitle("ElasticNet Coefficients per Cell Line", fontsize=16, y=0.98)
        fig.supxlabel("ElasticNet Coefficient", fontsize=14)
        fig.supylabel("Percentage", fontsize=14)
        plt.tight_layout()
        plt.show()

        # Find intersecting and unique columns
        intersecting_columns = heatmap_HepG2.columns.intersection(heatmap_K562.columns)
        unique_HepG2_columns = heatmap_HepG2.columns.difference(heatmap_K562.columns)
        unique_K562_columns = heatmap_K562.columns.difference(heatmap_HepG2.columns)

        # Subset to intersecting columns and cluster HepG2
        subset_HepG2 = heatmap_HepG2[intersecting_columns]
        linkage = sch.linkage(subset_HepG2.T, method="ward")
        dendrogram = sch.dendrogram(linkage, no_plot=True)
        intersecting_order = [subset_HepG2.columns[i] for i in dendrogram["leaves"]]

        # Cluster unique columns for each cell line
        linkage_HepG2_unique = sch.linkage(heatmap_HepG2[unique_HepG2_columns].T, method="ward")
        unique_HepG2_order = [heatmap_HepG2[unique_HepG2_columns].columns[i] for i in sch.dendrogram(linkage_HepG2_unique, no_plot=True)["leaves"]]

        linkage_K562_unique = sch.linkage(heatmap_K562[unique_K562_columns].T, method="ward")
        unique_K562_order = [heatmap_K562[unique_K562_columns].columns[i] for i in sch.dendrogram(linkage_K562_unique, no_plot=True)["leaves"]]

        # Final column orderings
        final_HepG2_order = intersecting_order + unique_HepG2_order
        final_K562_order = intersecting_order + unique_K562_order

        # Index of split point for plotting
        split_index_HepG2 = len(intersecting_order)
        split_index_K562 = len(intersecting_order)

        # Get global min and max values for consistent color scaling
        global_min = min(heatmap_HepG2.min().min(), heatmap_K562.min().min())
        global_max = max(heatmap_HepG2.max().max(), heatmap_K562.max().max())


        for plot_type in ["All", "Only Matching"]:

            if plot_type == "All":
                sharex = False 
            elif plot_type == "Only Matching":
                sharex = True
            # Plot heatmaps
            fig, axes = plt.subplots(2, 1, figsize=(30, 10), dpi=300, sharey=True, sharex=sharex, gridspec_kw={'height_ratios': [1, 1]})
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position for the single colorbar

            for ax, cell_line, heatmap, final_order, split_index in zip(
                            axes,
                            ["HepG2", "K562"],
                            [heatmap_HepG2, heatmap_K562],
                            [final_HepG2_order, final_K562_order],
                            [split_index_HepG2, split_index_K562],
            ):  
                                
                plotting = heatmap[final_order]

                if plot_type == "Only Matching":
                    plotting = plotting[intersecting_order]
                
                sns.heatmap(
                    plotting,
                    ax=ax,
                    cmap="seismic",
                    cbar=(ax == axes[0]),  # Add colorbar only for the first heatmap
                    cbar_ax=(cbar_ax if ax == axes[0] else None),
                    vmin=global_min,
                    vmax=global_max,
                    center=0.0,  # Set 0.0 as the center of the colorbar
                    xticklabels=True,
                    yticklabels=True,
                    linewidths=0.5,  # Thin black borders around cells
                    linecolor="black",
                )

                if plot_type == "All": 
                    ax.axvline(x=split_index, color="darkgreen", linewidth=7)
                
                ax.set_title(f"{cell_line}", fontsize=24, pad=10)
                ax.set_xlabel("")
                # Make y-axis tick labels larger
                ax.tick_params(axis='y', labelsize=24)
                ax.set_ylabel("")

            # Add colorbar title
            cbar_ax.set_title("Coef.", fontsize=25)
            cbar_ax.tick_params(labelsize=20)  # Increase tick label font size
            cbar_ax.set_box_aspect(20) 

            fig.supxlabel("RBP", fontsize=40, x=0.45)
            fig.supylabel("Position", fontsize=40, x=-0.001)

            plt.suptitle(f"ElasticNet Coefficients for {plot_type} Features", fontsize=50, y=1.06)
            plt.text(0.5, 1.3, "RBPs w/ eCLIP in both cell lines had 'Ward' clustering done in HepG2 and both cell lines data matches that ordering.\nRBPs that were unique to a cell line had their own 'Ward' clustering run to determine ordering.", ha='center', va='center', fontsize=20, transform=axes[0].transAxes)

            plt.tight_layout(rect=[0, 0, 0.91, 1])  # Adjust layout to make space for the colorbar
            plt.show()
            plt.close()

        # Scatterplot for matching features
        matching_features = []
        k562_coefficients = []
        hepg2_coefficients = []

        # Iterate through each RBP and position to ensure precise matching
        for position in heatmap_K562.index:
            for rbp in heatmap_K562.columns:
                if rbp in heatmap_HepG2.columns:
                    matching_features.append(f"{rbp}_{position}")
                    k562_coefficients.append(heatmap_K562.at[position, rbp])
                    hepg2_coefficients.append(heatmap_HepG2.at[position, rbp])

        # Convert to numpy arrays for correlation calculations
        k562_coefficients = np.array(k562_coefficients)
        hepg2_coefficients = np.array(hepg2_coefficients)

        pearson_corr, _ = pearsonr(k562_coefficients, hepg2_coefficients)
        spearman_corr, _ = spearmanr(k562_coefficients, hepg2_coefficients)

        # Create scatterplot
        plt.figure(figsize=(4.5,4), dpi=200)
        sns.scatterplot(
            x=k562_coefficients,
            y=hepg2_coefficients,
            alpha=0.5,
            edgecolor="black",
            color="deepskyblue",
            s=10  # Reduce the size of the dots
        )
        # Add y=x line
        plt.plot(
            [min(k562_coefficients), max(k562_coefficients)],
            [min(k562_coefficients), max(k562_coefficients)],
            color="red",
            linestyle="--",
            linewidth=1,
        )

        # Annotate coefficients with absolute values greater than 1.5
        for i, (k562, hepg2, feature) in enumerate(zip(k562_coefficients, hepg2_coefficients, matching_features)):
            if abs(k562) > .15 or abs(hepg2) > .15:
                plt.text(k562 - 0.02, hepg2+0.01, feature, fontsize=6, color="black", alpha=1)

        # Update title and text with smaller font and include total points
        total_points = len(k562_coefficients)
        plt.title(f"K562 vs HepG2 ElasticNet Coefficients\nNOTE: only includes matching features even though\nindividual models were run with different features", fontsize=10, y=1.04)
        plt.text(0.2, 0.9, f"Total Points: {total_points}\nPearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}", ha='center', va='center', fontsize=8, transform=plt.gca().transAxes)

        # Update axis labels with smaller font
        plt.xlabel("K562 Coefficients", fontsize=10)
        plt.ylabel("HepG2 Coefficients", fontsize=10)
        plt.tight_layout()
        plt.show()

    
    def get_has_RBP_KD_results(self, df, cell_line):

        if isinstance(df, pl.LazyFrame):
            df = df.select("index").collect()

        # Use the corrected_has_rbp_kd_dir with the cell line to glob for the file
        corrected_file_pattern = os.path.join(self.CORRECTED_HAS_RBP_KD_DIR, f"{cell_line}*.feather")
        corrected_files = glob.glob(corrected_file_pattern)
        assert len(corrected_files) == 1, f"Expected exactly one file for {cell_line}, but found {len(corrected_files)}"

        # Load the feather file
        corrected_df = pl.read_ipc(corrected_files[0])

        # Ensure the 'index' column exists in both dataframes
        assert "index" in df.columns, "'index' column is missing in the input dataframe"
        assert "index" in corrected_df.columns, "'index' column is missing in the corrected dataframe"

        # Perform a left join on the 'index' column
        joined_df = df.join(corrected_df, on="index", how="left")

        # Assert that all indices in df are in the corrected dataframe
        assert set(df["index"].to_list()).issubset(set(corrected_df["index"].to_list())), "Some indices in df are missing in the corrected dataframe"
        # Assert that the length of the dataframe has not changed after the join
        assert len(joined_df) == len(df), "The length of the dataframe changed after the join"
        # Assert that there are no null values in the joined dataframe
        assert joined_df.null_count().sum_horizontal().item() == 0, "Null values found in the joined dataframe"

        return joined_df


    def create_feature_metric_summary_table(self): 
        # Check if the summary table already exists
        if os.path.exists(self.CACHE_INFO["feature_metric_summary_table"]):
            logger.success("FROM CACHE: Feature metric summary table already exists.")
            
            self.feature_metric_summary_table = pd.read_csv(self.CACHE_INFO["feature_metric_summary_table"], sep="\t")
            return self.feature_metric_summary_table.head()

        else:
            logger.info("Feature metric summary table does not exist. Creating...")

            # Get global SHAP for 5_dfs_average
            global_shap_data = self.calculate_global_SHAP(mode="5_dfs_average")
            if not hasattr(self, 'elasticnet_info'):
                self.load_elasticnet_coefficients()

            summary_tables = []
            # Iterate through each cell line
            for cell_line in self.cell_lines:
                logger.info(f"Processing cell line: {cell_line}")
                
                # Initialize an empty list to store rows for the summary table
                summary_rows = []
                # Access global SHAP for the specific cell line
                global_shap = global_shap_data[cell_line]
                # Get ElasticNet coefficients
                elasticnet_coefficients = self.convert_RBP_position_to_2d_heatmap(self.elasticnet_info[cell_line]["coefficients"])

                # Ensure both dataframes have the same structure
                assert global_shap.index.equals(elasticnet_coefficients.index), "Index mismatch between global SHAP and ElasticNet coefficients"
                assert global_shap.columns.equals(elasticnet_coefficients.columns), "Column mismatch between global SHAP and ElasticNet coefficients"

                # Flatten the dataframes into long format
                for position in global_shap.index:
                    for rbp in global_shap.columns:
                        feature = f"{rbp}_{position}"
                        shap_value = global_shap.at[position, rbp]
                        coef_value = elasticnet_coefficients.at[position, rbp]
                        summary_rows.append({
                            "RBP": rbp,
                            "Position": position,
                            "Cell Line": cell_line,
                            "Feature": feature,
                            "Global SHAP": shap_value,
                            "ElasticNet Coefficient": coef_value,
                        })

                # Convert the list of rows into a DataFrame
                summary_table = pd.DataFrame(summary_rows).sort_values(by=["Cell Line", "Feature"])
            
                xgboost_metadata = self.hash_metadata[
                    (self.hash_metadata["cell_line"] == cell_line) & 
                    (self.hash_metadata["name"] == "XGBRegressor")
                ].sort_values(by="hash")
                
                # Take the first hash
                first_hash = xgboost_metadata.iloc[0]["hash"]
                
                # Load the corresponding SHAP file
                shap_file = f"{self.SHAP_DIR}/{first_hash}.feather"

                # Scan the SHAP file and collect the schema names
                df = pl.scan_ipc(shap_file)
                schema = df.collect_schema().names()

                logger.info("Calculating Binding Percentage for each feature")
                # Subset to columns that end in "_binding" and take the sum of those columns
                binding_columns = [col for col in schema if col.endswith("_binding")]
                binding_sum = df.select(pl.col(binding_columns).sum()).collect()
                num_rows = df.select(pl.count()).collect().item()
                
                # Calculate binding percentage
                binding_percentage = binding_sum / num_rows

                # Transpose binding_percentage and reset its index
                binding_percentage_long = binding_percentage.to_pandas().transpose().reset_index()
                binding_percentage_long.columns = ["Feature", "Binding Percentage"]
                # Remove the "_binding" suffix from the Feature column in binding_percentage_long
                binding_percentage_long["Feature"] = binding_percentage_long["Feature"].str.replace("_binding", "", regex=False)

                summary_table_length_original = len(summary_table)
                # Ensure a 1-to-1 inner merge with the summary table
                summary_table = summary_table.merge(binding_percentage_long, how="inner", on="Feature")
                assert len(summary_table) == summary_table_length_original, "Inner merge resulted in a different number of rows"
                # Assert that there are no null values in the merged summary_table
                assert summary_table.notnull().all().all(), "Null values found in summary_table after merge"
                
                logger.info("Calculating # differential significant events")

                differential_df = df.filter(
                    (pl.col("FDR") <= self.FDR_THRESHOLD) &
                    (pl.col("DeltaPSI").abs() >= self.DPSI_THRESHOLD) &
                    (pl.col("RBP_KD_Target") != "CTRL")
                ).select(
                    ["index", "RBP_KD_Target", "rMATS Event ID"]
                ).unique(
                    subset = ["RBP_KD_Target", "rMATS Event ID"]
                ).collect()
                
                has_RBP_KD_df = self.get_has_RBP_KD_results(df, cell_line)
                # Perform a left join on the "index" column
                joined_df = differential_df.join(has_RBP_KD_df, on="index", how="left")

                assert len(joined_df) == len(differential_df), "The length of the joined dataframe changed after the join"
                assert joined_df.null_count().sum_horizontal().item() == 0, "Null values found in the joined dataframe"
            
                for rbp in summary_table["RBP"].unique():
                    num_diff_events = len(joined_df.filter(pl.col("RBP_KD_Target") == rbp))
                    num_diff_events_with_rbp_kd = len(
                        joined_df.filter((pl.col("RBP_KD_Target") == rbp) & (pl.col("has_RBP_KD") == True))
                    )
                    summary_table.loc[summary_table["RBP"] == rbp, "# RBP Diff. Events"] = num_diff_events
                    summary_table.loc[summary_table["RBP"] == rbp, "# RBP Diff. Events w/ RBP KD"] = num_diff_events_with_rbp_kd

                    for position in range(1,7): 
                        feature = f"{rbp}_{position}"
                        num_diff_events_with_rbp_kd_in_position = len(
                            joined_df.filter(
                                (pl.col("RBP_KD_Target") == rbp) & 
                                (pl.col(f"has_RBP_KD_{position}") == True) 
                            )
                        )        

                        summary_table.loc[summary_table["Feature"] == feature, "# RBP Diff. Events w/ RBP KD at Position"] = num_diff_events_with_rbp_kd_in_position        
                
                # Add the summary table to the list
                summary_tables.append(summary_table)

            # Concatenate all summary tables for each cell line
            summary_table = pd.concat(summary_tables, ignore_index=True)
            # Sort the summary table by Cell Line and Feature
            summary_table = summary_table.sort_values(by=["Cell Line", "Feature"])
            # Reset the index of the summary table
            summary_table.reset_index(drop=True, inplace=True)

            # Save the summary table to a TSV file
            summary_table.to_csv(self.CACHE_INFO["feature_metric_summary_table"], sep="\t", index=False)
            logger.success("Created feature metric summary table")
            return summary_table


    def plot_feature_metric_summary_table(self):
        if not hasattr(self, 'feature_metric_summary_table'):
            self.create_feature_metric_summary_table()
        
        for plotting_column in ["# RBP Diff. Events", "# RBP Diff. Events w/ RBP KD", "# RBP Diff. Events w/ RBP KD at Position"]:
            logger.info(f"Plotting {plotting_column} for each cell line")
            for mode in ["All Features", "Only RBPs"]:

                if plotting_column == "# RBP Diff. Events w/ RBP KD at Position" and mode == "Only RBPs":
                    logger.info(f"Skipping {plotting_column} for {mode} as it is not applicable")
                    continue

                logger.info(f"Plotting for mode: {mode}")

                fig, axes = plt.subplots(3, len(self.cell_lines), figsize=(13, 13), dpi=300, sharey=True, sharex="row")
                row_colors = ["lightcoral", "lightgreen", "lightskyblue"]

                for col_idx, cell_line in enumerate(self.cell_lines):
                    # Subset to the specific cell line
                    data = self.feature_metric_summary_table[self.feature_metric_summary_table["Cell Line"] == cell_line]
                    data = data.copy()
                    data["Abs(ElasticNet Coefficient)"] = data["ElasticNet Coefficient"].abs()
                    
                    logger.warning("REMINDER: multiplying 'Binding Percentage' by 100 due to mistake in not originally doing so")
                    data["Binding Percentage"] *= 100

                    if mode == "Only RBPs":
                        # Group by RBP and aggregate
                        data = data.groupby("RBP", as_index=False).agg({
                            "Global SHAP": "mean",
                            "Abs(ElasticNet Coefficient)": "mean",
                            "Binding Percentage": "mean",
                            plotting_column: "mean"
                        })

                    # Iterate over the x-axis columns to compare against plotting_column
                    for row_idx, x_col in enumerate(["Global SHAP", "Abs(ElasticNet Coefficient)", "Binding Percentage"]):
                        ax = axes[row_idx, col_idx]
                        sns.scatterplot(
                            data=data,
                            x=x_col,
                            y=plotting_column,
                            alpha=0.7,
                            edgecolor="black",
                            color=row_colors[row_idx],
                            ax=ax
                        )
                        ax.set_title(f"{cell_line}: {x_col}", fontsize=14)
                        ax.set_xlabel(x_col, fontsize=12, color = row_colors[row_idx], alpha=1)
                        ax.set_ylabel("")
                        ax.tick_params(axis="both", labelsize=12)

                        # Label the top 10 values on both the x-axis and the plotting_column axis
                        top_10_x = data.nlargest(10, x_col)
                        top_10_y = data.nlargest(10, plotting_column)
                        top_10_combined = pd.concat([top_10_x, top_10_y]).drop_duplicates()

                        for _, row in top_10_combined.iterrows():
                            if mode == "Only RBPs":
                                text = row["RBP"]
                            elif mode == "All Features":
                                text = row["Feature"]
                            ax.text(
                                row[x_col],
                                row[plotting_column]+30,
                                text,
                                fontsize=6,
                                color="black",
                                alpha=0.8
                            )

                        # Add Pearson, Spearman, and number of points in the center right of the plot
                        pearson_corr, _ = pearsonr(data[x_col], data[plotting_column])
                        spearman_corr, _ = spearmanr(data[x_col], data[plotting_column])
                        num_points = len(data)

                        ax.text(
                            0.98, 0.5,
                            f"Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}\nPoints: {num_points}",
                            transform=ax.transAxes,
                            fontsize=12,
                            verticalalignment='center',
                            horizontalalignment='right'
                        )

                if mode == "Only RBPs":
                    suffix = "- X-axis RBP value comes from average of 6 positions"
                elif mode == "All Features":
                    suffix = ""

                plt.suptitle(
                    f"XGBoost, Linear Model, Binding vs\nSignificant {plotting_column} (FDR ≤ {self.FDR_THRESHOLD}, ΔPSI ≥ {self.DPSI_THRESHOLD})\n NOTE: {mode} {suffix}", 
                    fontsize=20, y=1.02
                )
                fig.supylabel(
                    f"{plotting_column}", fontsize=20, x=-0.02
                )
                fig.supxlabel(
                    f"Cell Line", fontsize=20
                )
                # plt.tight_layout(rect=[0, 0, 1, 0.9])
                plt.tight_layout()
                plt.show()


    def plot_differential_splicing_types(self): 

        if not hasattr(self, 'feature_metric_summary_table'):
            self.create_feature_metric_summary_table()

        plotting_columns_info = {
            "# RBP Diff. Events": "RBP-specific",
            "# RBP Diff. Events w/ RBP KD": "RBP-specific",
            "# RBP Diff. Events w/ RBP KD at Position": "Feature-specific"
        }

        for plotting_column, specificity in plotting_columns_info.items():
            logger.info(f"Processing {plotting_column} ({specificity})")

            # Deep copy the feature metric summary table
            metric_summary_table = self.feature_metric_summary_table.copy()
            # Assert that there are no null or missing values in the metric_summary_table
            assert metric_summary_table.notnull().all().all(), "Null or missing values found in the metric_summary_table"

            if specificity == "RBP-specific":
                metric_summary_table = metric_summary_table.drop_duplicates(subset=["Cell Line", "RBP"], keep="first")
            elif specificity == "Feature-specific":
                pass  # Use the entire table

            # # First Figure: Histogram and Boxplot
            fig, axes = plt.subplots(len(self.cell_lines), 2, figsize=(8, 5), dpi=300, sharex=True, sharey="col")
            for row_idx, cell_line in enumerate(self.cell_lines):
                cell_line_data = metric_summary_table[metric_summary_table["Cell Line"] == cell_line]

                # Histogram
                sns.histplot(
                    cell_line_data[plotting_column],
                    bins=50,
                    stat="percent",
                    color="deepskyblue",
                    edgecolor="black",
                    alpha=0.7,
                    ax=axes[row_idx, 0]
                )
                axes[row_idx, 0].set_title(f"{cell_line} Histogram", fontsize=10)
                axes[row_idx, 0].text(
                    0.5, 0.95, f"n={len(cell_line_data)}",
                    transform=axes[row_idx, 0].transAxes,
                    fontsize=8, verticalalignment="top"
                )

                # Boxplot with Stripplot
                sns.boxplot(
                    data=cell_line_data[plotting_column],
                    orient="h",
                    color="deepskyblue",
                    ax=axes[row_idx, 1],
                    width=0.5,
                    showmeans=True,
                    meanline=True,
                    meanprops={"color": "red", "linewidth": 1.5}
                )
                sns.stripplot(
                    data=cell_line_data[plotting_column],
                    orient="h",
                    color="black",
                    size=5,
                    alpha=0.2,
                    ax=axes[row_idx, 1]
                )
                axes[row_idx, 1].set_title(f"{cell_line} Boxplot", fontsize=10)
                axes[row_idx, 1].text(
                    0.5, 0.95, f"n = {len(cell_line_data)}",
                    transform=axes[row_idx, 1].transAxes,
                    fontsize=8, verticalalignment="top"
                )

                # Label the top 5 points in the stripplot
                top_5 = cell_line_data.nlargest(5, plotting_column)
                for _, row in top_5.iterrows():
                    axes[row_idx, 1].text(
                        row[plotting_column],  # Adjust the x-coordinate for better visibility
                        0,  # Use the index or a unique identifier for labeling
                        row["RBP"] if specificity == "RBP-specific" else row["Feature"],
                        fontsize=6,
                        color="green",
                        alpha=0.8
                    )
                
                # Calculate the percentage of values above the threshold
                threshold = 0  # Set your desired threshold here
                percentage_above_threshold = (cell_line_data[plotting_column] > threshold).mean() * 100

                # Add the percentage to the bottom right corner of the plot
                axes[row_idx, 1].text(
                    0.95, 0.05,
                    f"% Values Above {threshold}: {percentage_above_threshold:.2f}%",
                    transform=axes[row_idx, 1].transAxes,
                    fontsize=8,
                    verticalalignment="bottom",
                    horizontalalignment="right", 
                    color = "chocolate"
                )
                
                axes[row_idx, 0].set_xlabel("")
                axes[row_idx, 1].set_xlabel("")
                
            fig.supxlabel(f"{plotting_column}", fontsize=11)
            fig.supylabel(f"Cell Line", fontsize=11, x=0.01)
            plt.suptitle(f"Significant {plotting_column}\nNOTE: this is {specificity}", fontsize=14, y=0.99)
            plt.tight_layout()
            plt.show()

            # Second Figure: Heatmap
            if specificity == "RBP-specific":
                y_fig_size = 7
            elif specificity == "Feature-specific":
                y_fig_size = 12
            
            for log_transform in [False, True]:
                fig, axes = plt.subplots(2, 1, figsize=(30, y_fig_size), dpi=300)

                for ax, cell_line in zip(axes, self.cell_lines):
                    # Subset data for the specific cell line
                    cell_line_data = metric_summary_table[metric_summary_table["Cell Line"] == cell_line]

                    if specificity == "RBP-specific":
                        heatmap_data = cell_line_data.pivot(index="Cell Line", columns="RBP", values=plotting_column)
                    elif specificity == "Feature-specific":
                        heatmap_data = cell_line_data.pivot(index="Position", columns="RBP", values=plotting_column)
                    assert heatmap_data.notnull().all().all(), "Null or missing values found in the heatmap data"

                    if specificity == "Feature-specific":
                        # Perform hierarchical clustering
                        linkage = sch.linkage(heatmap_data.T, method="ward")
                        dendrogram = sch.dendrogram(linkage, no_plot=True)
                        ordered_columns = [heatmap_data.columns[i] for i in dendrogram["leaves"]]
                    else:
                        # Sort columns by the sum of their actual heatmap values
                        ordered_columns = heatmap_data.sum(axis=0).sort_values().index.tolist()

                    heatmap_data = heatmap_data[ordered_columns]

                    # Define the norm for the colorbar scale
                    norm = LogNorm() if log_transform else None

                    # Plot heatmap for the cell line
                    sns.heatmap(
                        heatmap_data,
                        ax=ax,
                        cmap="viridis",
                        cbar=True,
                        linewidths=0.5,
                        linecolor="gray",
                        cbar_kws={"shrink": 0.8, "aspect": 5, "pad": 0.01},  # Adjust colorbar position and size
                        norm=norm  # Apply log scale to the colorbar if specified
                    )
                    cbar = ax.collections[0].colorbar  # Get the colorbar
                    cbar.ax.tick_params(labelsize=12)  # Set the font size of the colorbar ticks

                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    if specificity == "Feature-specific":
                        ax.set_title(f"{cell_line}", fontsize=20)
                    ax.tick_params(axis='y', labelsize=24)

                fig.supxlabel("RBP", fontsize=30, x=0.45)
                fig.supylabel("Cell Line" if specificity == "RBP-specific" else "Position", fontsize=30, x=-0.0001)

                if specificity == "Feature-specific":
                    suffix = "\nNOTE 2: Ward hierarchical clustering run for each cell line for ordering RBPs."
                elif specificity == "RBP-specific":
                    suffix = ""

                transform_label = "(Log Transformed)" if log_transform else "(No Log Transform)"
                plt.suptitle(f"{transform_label} Significant {plotting_column} Heatmap\nNOTE: this is {specificity}{suffix}", fontsize=30, y=1.02)
                plt.tight_layout()
                plt.show()

            # Third Figure: Matching Features Heatmap
            matching_data = {}

            if specificity == "RBP-specific":
                col_to_select = "RBP"
            elif specificity == "Feature-specific":
                col_to_select = "Feature"
            
            # Get unique per cell line
            unique_per_cell_line = {
                cell_line: set(metric_summary_table[metric_summary_table["Cell Line"] == cell_line][col_to_select].unique())
                for cell_line in self.cell_lines
            }
            # Find the intersection across both cell lines
            intersecting = unique_per_cell_line[self.cell_lines[0]].intersection(unique_per_cell_line[self.cell_lines[1]])
            summary_table = metric_summary_table[metric_summary_table[col_to_select].isin(intersecting)]

            if specificity == "Feature-specific":
                # Create a pivot table for the heatmap
                pivot_table = summary_table[summary_table["Cell Line"] == self.cell_lines[0]].pivot(
                    index="Position",
                    columns="RBP",
                    values=plotting_column
                )
                assert pivot_table.notnull().all().all(), "Null or missing values found in the pivot table"

                # Perform hierarchical clustering on the pivot table
                linkage = sch.linkage(pivot_table.T, method="ward")
                dendrogram = sch.dendrogram(linkage, no_plot=True)
                ordered_features = [pivot_table.columns[i] for i in dendrogram["leaves"]]

                # Create a new pivot table with the ordered features
                matching_data = {
                    cell_line: summary_table[summary_table["Cell Line"] == cell_line].pivot(
                        index="Position",
                        columns="RBP",
                        values=plotting_column
                    ).reindex(columns=ordered_features)
                    for cell_line in self.cell_lines
                }

            elif specificity == "RBP-specific":
                ordered_features = summary_table[summary_table["Cell Line"] == self.cell_lines[0]].sort_values(by=plotting_column)[col_to_select].tolist()
                
                matching_data = {
                    cell_line: summary_table[summary_table["Cell Line"] == cell_line].pivot( 
                        index= "Cell Line", 
                        columns=col_to_select,
                        values=plotting_column
                    ).reindex(columns = ordered_features)
                    for cell_line in self.cell_lines
                }

            # Determine global min and max values for consistent color scaling
            vmin = min(matching_data[cell_line].min().min() for cell_line in self.cell_lines)
            vmax = max(matching_data[cell_line].max().max() for cell_line in self.cell_lines)

            for log_transform in [False, True]:
                fig, axes = plt.subplots(2, 1, figsize=(30, y_fig_size), dpi=300, sharex=True)
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position for the single colorbar

                # Define a single norm for both cell lines
                norm = LogNorm(vmin=vmin+1, vmax=vmax) if log_transform else None

                for ax, cell_line in zip(axes, self.cell_lines):
                    sns.heatmap(
                        matching_data[cell_line],
                        cmap="viridis",
                        cbar=(ax == axes[0]),  # Add colorbar only for the first heatmap
                        cbar_ax=(cbar_ax if ax == axes[0] else None),
                        linewidths=0.5,
                        linecolor="gray",
                        norm=norm,  # Use the shared norm for both heatmaps
                        vmin= vmin if not log_transform else None,  # Explicitly pass vmin
                        vmax=vmax if not log_transform else None,  # Explicitly pass vmax
                        ax=ax
                    )
                    ax.set_title(f"{cell_line} {'Ordered' if cell_line == self.cell_lines[0] else 'Matching'}", fontsize=30)
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.tick_params(axis='y', labelsize=24)

                # Add colorbar title
                cbar_ax.tick_params(labelsize=20)
                cbar_ax.set_box_aspect(20)  # Make the colorbar skinnier

                fig.supxlabel("RBP", fontsize=30, x=0.45)
                fig.supylabel("Cell Line" if specificity == "RBP-specific" else "Position", fontsize=30, x=-0.0001)

                if specificity == "Feature-specific":
                    suffix = f"\nNOTE 2: Ward hierarchical clustering run in {self.cell_lines[0]} for ordering RBPs."
                elif specificity == "RBP-specific":
                    suffix = ""

                transform_label = "(Log Transformed)" if log_transform else "(No Log Transform)"
                plt.suptitle(f"{transform_label} Significant {plotting_column} for Matching\nNOTE: this is {specificity}{suffix}", fontsize=30, y=1.02)
                plt.tight_layout(rect=[0, 0, 0.91, 1])  # Adjust layout to make space for the colorbar
                plt.show()

            # Fourth Figure: Scatterplot
            
            # Convert matching_data to long-form tables
            long_form_0 = matching_data[self.cell_lines[0]].stack().reset_index()
            long_form_1 = matching_data[self.cell_lines[1]].stack().reset_index()

            # Rename columns for clarity
            if specificity == "Feature-specific":
                long_form_0.columns = ["Position", "RBP", f"{self.cell_lines[0]} {plotting_column}"]
                long_form_1.columns = ["Position", "RBP", f"{self.cell_lines[1]} {plotting_column}"]

                # Merge the two tables on RBP and Position
                merged_data = pd.merge(long_form_0, long_form_1, on=["Position", "RBP"])

            elif specificity == "RBP-specific":
                long_form_0.columns = ["Cell Line", "RBP", f"{self.cell_lines[0]} {plotting_column}"]
                long_form_1.columns = ["Cell Line", "RBP", f"{self.cell_lines[1]} {plotting_column}"]

                # Merge the two tables on RBP
                merged_data = pd.merge(long_form_0, long_form_1, on=["RBP"])
            
            assert merged_data.notnull().all().all(), "Null or missing values found in the merged data"
            assert len(merged_data) == len(long_form_0), "The length of the merged data is not equal to the length of long_form_0"

            # Calculate correlations
            pearson_corr, _ = pearsonr(
                merged_data[f"{self.cell_lines[0]} {plotting_column}"],
                merged_data[f"{self.cell_lines[1]} {plotting_column}"]
            )
            spearman_corr, _ = spearmanr(
                merged_data[f"{self.cell_lines[0]} {plotting_column}"],
                merged_data[f"{self.cell_lines[1]} {plotting_column}"]
            )

            # Plot scatterplot
            plt.figure(figsize=(5, 4), dpi=200)
            sns.scatterplot(
                data=merged_data,
                x=f"{self.cell_lines[0]} {plotting_column}",
                y=f"{self.cell_lines[1]} {plotting_column}",
                alpha=0.7,
                edgecolor="black",
                color="deepskyblue",
                s=20
            )

            # Add y=x line
            plt.plot(
                [merged_data[f"{self.cell_lines[0]} {plotting_column}"].min(), merged_data[f"{self.cell_lines[0]} {plotting_column}"].max()],
                [merged_data[f"{self.cell_lines[0]} {plotting_column}"].min(), merged_data[f"{self.cell_lines[0]} {plotting_column}"].max()],
                color="red",
                linestyle="--",
                linewidth=1,
                label="y=x"
            )

            # Annotate top 5 highest values in both columns
            top_5_col1 = merged_data.nlargest(10, f"{self.cell_lines[0]} {plotting_column}")
            top_5_col2 = merged_data.nlargest(10, f"{self.cell_lines[1]} {plotting_column}")
            top_5_combined = pd.concat([top_5_col1, top_5_col2]).drop_duplicates()

            for _, row in top_5_combined.iterrows():
                if specificity == "RBP-specific":
                    text = row["RBP"]
                elif specificity == "Feature-specific":
                    text = f"{row['RBP']}_{row['Position']}"
                plt.text(
                    row[f"{self.cell_lines[0]} {plotting_column}"] - 20,
                    row[f"{self.cell_lines[1]} {plotting_column}"] + 20,
                    text,
                    fontsize=4,
                    color="black",
                    alpha=0.8
                )

            plt.title(f"Significant {plotting_column} for Matching\n({specificity})", fontsize=12, y=1.01)
            plt.xlabel(f"{self.cell_lines[0]}", fontsize=12)
            plt.ylabel(f"{self.cell_lines[1]}", fontsize=12)
            plt.text(
                0.75, 0.97,
                f"Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}\nPoints: {len(merged_data)}",
                transform=plt.gca().transAxes,
                fontsize=8,
                verticalalignment="top"
            )
            plt.tight_layout()
            plt.show()


    def plot_differential_ranks(self): 
        if not hasattr(self, 'feature_metric_summary_table'):
            self.create_feature_metric_summary_table()

        # Copy the feature metric summary table
        summary_table = self.feature_metric_summary_table.copy()
        # Take unique values by both Cell Line and RBP
        summary_table = summary_table.drop_duplicates(subset=["Cell Line", "RBP"])

        for cell_line in self.cell_lines:
            # Subset the data for the specific cell line
            cell_line_data = summary_table[summary_table["Cell Line"] == cell_line].copy()

            # Rank the columns of interest
            cell_line_data["Rank: # RBP Diff. Events"] = cell_line_data["# RBP Diff. Events"].rank(ascending=False)
            cell_line_data["Rank: # RBP Diff. Events w/ RBP KD"] = cell_line_data["# RBP Diff. Events w/ RBP KD"].rank(ascending=False)

            # Calculate the change in ranks
            cell_line_data["Rank Change"] = abs(cell_line_data["Rank: # RBP Diff. Events"] - cell_line_data["Rank: # RBP Diff. Events w/ RBP KD"])

            # Scatterplot of the ranks
            plt.figure(figsize=(6,4), dpi=300)
            sns.scatterplot(
                data=cell_line_data,
                x="Rank: # RBP Diff. Events",
                y="Rank: # RBP Diff. Events w/ RBP KD",
                alpha=0.7,
                edgecolor="black",
                color="deepskyblue",
                s=20
            )

            # Add a diagonal line for reference
            plt.plot(
                [cell_line_data["Rank: # RBP Diff. Events"].min(), cell_line_data["Rank: # RBP Diff. Events"].max()],
                [cell_line_data["Rank: # RBP Diff. Events"].min(), cell_line_data["Rank: # RBP Diff. Events"].max()],
                color="red",
                linestyle="--",
                linewidth=1,
                label="y=x"
            )

            # Annotate the top 5 rank changes
            top_5 = cell_line_data.nlargest(15, "Rank Change")
            for _, row in top_5.iterrows():
                plt.text(
                    row["Rank: # RBP Diff. Events"]-2,
                    row["Rank: # RBP Diff. Events w/ RBP KD"]+2,
                    row["RBP"],
                    fontsize=2,
                    color="black",
                    alpha=0.8
                )

            # Calculate Pearson and Spearman correlations
            pearson_corr, _ = pearsonr(
                cell_line_data["Rank: # RBP Diff. Events"],
                cell_line_data["Rank: # RBP Diff. Events w/ RBP KD"]
            )
            spearman_corr, _ = spearmanr(
                cell_line_data["Rank: # RBP Diff. Events"],
                cell_line_data["Rank: # RBP Diff. Events w/ RBP KD"]
            )

            # Add labels, title, and legend
            plt.title(f"Rank Comparison for {cell_line} (Scatterplot)", fontsize=14)
            plt.xlabel("Rank: # RBP Diff. Events", fontsize=12)
            plt.ylabel("Rank: # RBP Diff. Events w/ RBP KD", fontsize=12)

            # Add correlation and point count in the bottom right corner
            num_points = len(cell_line_data)
            plt.text(
                0.97, 0.03,
                f"Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}\nPoints: {num_points}",
                transform=plt.gca().transAxes,
                fontsize=8,
                verticalalignment='bottom',
                horizontalalignment='right'
            )

            plt.tight_layout()
            plt.show()

            # Lineplot of the ranks
            plt.figure(figsize=(6,4), dpi=300)
            sorted_data = cell_line_data.sort_values("Rank: # RBP Diff. Events")

            # Plot a separate line for each row
            for _, row in sorted_data.iterrows():
                plt.plot(
                    [row["Rank: # RBP Diff. Events"], row["Rank: # RBP Diff. Events"]],
                    [row["Rank: # RBP Diff. Events"], row["Rank: # RBP Diff. Events w/ RBP KD"]],
                    marker="o",
                    linestyle="-",
                    color="deepskyblue",
                    alpha=0.7
                )

            # Annotate the top 5 rank changes
            for _, row in top_5.iterrows():
                plt.text(
                    row["Rank: # RBP Diff. Events"]-1,
                    row["Rank: # RBP Diff. Events w/ RBP KD"]+2,
                    row["RBP"],
                    fontsize=4,
                    color="black",
                    alpha=0.8
                )

            # Add labels, title, and legend
            plt.title(f"Rank Comparison for {cell_line} (Lineplot)", fontsize=14)
            plt.xlabel("Rank: # RBP Diff. Events", fontsize=12)
            plt.ylabel("Rank: # RBP Diff. Events w/ RBP KD", fontsize=12)
            plt.tight_layout()
            plt.show()


    def tmp(self): 

        rbp_counts = {}

        for file in sorted(glob.glob("../../../../../../data/collaborators/BWH/1_ENCODE_shRNA_RBP_KD_2024-04-hg38-gencode-v29/**/SE.*", recursive=True)):
            if "HepG2" in file and "Transfection" not in file: 
                rbp = file.split("/")[-2].split("-")[0]

                df = pl.scan_csv(file, separator="\t")
                filtered_df = df.filter(
                    (pl.col("FDR") <= self.FDR_THRESHOLD) & 
                    (pl.col("IncLevelDifference").abs() >= self.DPSI_THRESHOLD)
                )
                count = filtered_df.collect().height
                rbp_counts[rbp] = count

        rbp_counts_df = pd.DataFrame(list(rbp_counts.items()), columns=["RBP", "Event Count"])
        summary_table = self.feature_metric_summary_table.copy()
        summary_table = summary_table[summary_table["Cell Line"] == "HepG2"]
        summary_table = summary_table.merge(rbp_counts_df, how="left", left_on="RBP", right_on="RBP")
        summary_table = summary_table.dropna()

        # Take unique rows of summary_table by RBP and the two columns used later
        summary_table = summary_table[["RBP", "# RBP Differential Events", "Event Count"]].drop_duplicates(subset=["RBP"])

        # Calculate correlation values
        pearson_corr, _ = pearsonr(summary_table["# RBP Differential Events"], summary_table["Event Count"])
        spearman_corr, _ = spearmanr(summary_table["# RBP Differential Events"], summary_table["Event Count"])

        # Plot # RBP Differential Events vs event count as a scatterplot
        plt.figure(figsize=(6, 4), dpi=200)
        sns.scatterplot(
            data=summary_table,
            x="# RBP Differential Events",
            y="Event Count",
            alpha=0.7,
            edgecolor="black",
            color="deepskyblue",
            s=20
        )

        # Add a diagonal line for reference
        plt.plot(
            [summary_table["# RBP Differential Events"].min(), summary_table["# RBP Differential Events"].max()],
            [summary_table["# RBP Differential Events"].min(), summary_table["# RBP Differential Events"].max()],
            color="red",
            linestyle="--",
            linewidth=1,
            label="y=x"
        )

        # Annotate the top 6 values in "Event Count"
        top_6 = summary_table.nlargest(10, "Event Count")
        for _, row in top_6.iterrows():
            plt.text(
                row["# RBP Differential Events"] + -100, 
                row["Event Count"]+100, 
                row["RBP"], 
                fontsize=6, 
                color="black"
            )

        # Add labels, title, and correlation values
        plt.title("Yogi's Data vs rMATS Original File # Diff Events", fontsize=12)
        plt.xlabel("Yogi's Data # Diff Events", fontsize=10)
        plt.ylabel("rMATS Original File # Diff Events", fontsize=10)
        num_points = len(summary_table)
        plt.text(
            0.05, 0.85, 
            f"Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}\nPoints: {num_points}", 
            transform=plt.gca().transAxes, 
            fontsize=8, 
            verticalalignment='top'
        )
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
        plt.close()

        return summary_table