import glob, os, json, gc, pickle, gzip, tempfile, shutil, tqdm

import pandas as pd, polars as pl, numpy as np, matplotlib.pyplot as plt, seaborn as sns

from dataclasses import dataclass
from IPython.display import display, Video
from loguru import logger
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

@dataclass
class ShapNetworkInvestigator:
    PARAMS_DIR = "../../3_choose_dataset_and_model_parameters/output/model_reproduction/model_parameters/"
    SHAP_MODEL_PICKLE_DIR = "../../4_run_final_models_and_SHAP/outputs/SHAP/regular/normal/explainer_objects/"
    SHAP_DIR = "../../4_run_final_models_and_SHAP/outputs/SHAP/regular/normal/shap_values/"
    SHAP_TYPE = "regular-observational"

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
            }
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
        assert all((df["index"].to_numpy() == cell_line_shap[0]["index"].to_numpy()).all() for df in cell_line_shap), "Index order mismatch across SHAP DataFrames"
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

        # Extract RBP and position information from column names
        rbp_positions = [self.get_RBP_position(col) for col in table.columns]

        # Create a DataFrame with RBP and position as separate columns
        rbp_positions_df = pd.DataFrame(
            [(rbp, position, table[0, col]) for (rbp, position), col in zip(rbp_positions, table.columns)],
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
            if mode == '5_dfs':
                dfs_global_SHAP = {
                    cell_line: self.retrieve_5_SHAP_tables_per_cell_line(cell_line) for cell_line in self.cell_lines
                }

                for cell_line, dfs in dfs_global_SHAP.items():
                    # Drop the 'index' column and calculate the absolute value average of each column
                    averaged_df = [df.sort('index').drop('index').select(pl.all().abs().mean()) for df in dfs]
                    # Convert each averaged DataFrame to a 2D heatmap
                    heatmaps = [self.convert_RBP_position_to_2d_heatmap(df) for df in averaged_df]
                    # Store the heatmaps in the global dictionary
                    global_heatmaps[cell_line] = heatmaps

            elif mode == '5_dfs_average':
                raise NotImplementedError("This mode is not implemented yet")
            
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
            fig, axes = plt.subplots(2, 1, figsize=(10, 16), dpi=200)
            fig.suptitle("Global SHAP Heatmaps (5_dfs_average)", fontsize=16)

            for ax, (cell_line, heatmap) in zip(axes, global_SHAP.items()):
                sns.heatmap(
                    heatmap,
                    ax=ax,
                    cmap="viridis",
                    cbar=True
                )
                ax.set_title(f"Cell Line: {cell_line}")
                ax.set_xlabel("RBP")
                ax.set_ylabel("Position")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()


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


    def tmp(self): 
        
        # self.shap_dfs = self.retrieve_5_SHAP_tables_per_cell_line("K562")
        max_values = [df.select(pl.all().exclude("index").max()).to_numpy().max() for df in self.shap_dfs]
        min_values = [df.select(pl.all().exclude("index").min()).to_numpy().min() for df in self.shap_dfs]
        logger.info(f"Max Values: {max_values}")
        logger.info(f"Min Values: {min_values}")
        overall_max = max(max_values)
        overall_min = min(min_values)
        logger.info(f"Overall Max Value: {overall_max}, Overall Min Value: {overall_min}")