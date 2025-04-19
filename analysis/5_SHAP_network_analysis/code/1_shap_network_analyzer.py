import glob, os, json, gc, pickle

import pandas as pd, polars as pl, numpy as np, matplotlib.pyplot as plt, seaborn as sns

from dataclasses import dataclass
from loguru import logger


@dataclass
class ShapNetworkInvestigator:
    PARAMS_DIR = "../../3_choose_dataset_and_model_parameters/output/model_reproduction/model_parameters/"
    SHAP_DIR = "../../4_run_final_models_and_SHAP/outputs/SHAP/regular/normal/shap_values/"
    SHAP_TYPE = "regular-observational"

    CACHE_INFO = {
            "hash_metadata": "../outputs/hash_metadata/hash_metadata.tsv",
            "SHAP_std": {
                "K562": "../outputs/SHAP_std/K562_SHAP_std.feather",
                "HepG2": "../outputs/SHAP_std/HepG2_SHAP_std.feather",
            }, 
            "global_SHAP": {
                "5_dfs": "../outputs/global_SHAP/5_dfs.pkl", 
                "5_dfs_average": "../outputs/global_SHAP/5_dfs_average.pkl",
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

            hash_metadata = pd.DataFrame(data)
            hash_metadata = hash_metadata.set_index('hash').reset_index().sort_values(by=['name', 'cell_line'])
            # Save to CSV
            hash_metadata.to_csv(f"{HASH_FILE}", sep="\t", index=False)
            logger.success(f"Created hash metadata file")
            
        self.hash_metadata = hash_metadata

    
    def calculate_pointwise_SHAP_metric_per_cell_line(self, cell_line_shap= None, metric = None, output_file = None):
        
        assert None not in (cell_line_shap, metric, output_file), "Arguments 'data', 'metric', and 'output_file' cannot be None"
        logger.info(f"Calculating pointwise SHAP metric -- {metric} -- and saving to {output_file}")

        # Ensure we have 5 dataframes
        assert len(cell_line_shap) == 5

        # Sort each DataFrame in cell_line_shap by the "index" column
        cell_line_shap = [df.sort("index").drop('index') for df in cell_line_shap]

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
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Convert the result back to a Polars DataFrame
        result_df = pl.DataFrame(result, schema=cell_line_shap[0].columns)
        # Assert that the result and result_df have the same number of columns and rows as the first DataFrame in cell_line_shap
        assert result.shape == cell_line_shap[0].shape, "Resulting metric array has inconsistent dimensions"
        assert result_df.shape == cell_line_shap[0].shape, "Resulting DataFrame has inconsistent dimensions"

        # Save the result to the specified output file
        result_df.write_ipc(output_file)
        logger.success(f"Saved pointwise SHAP {metric} for cell line to {output_file}")

    
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


    def calculate_global_SHAP(self, mode=None): 
        assert mode in ['5_dfs', '5_dfs_average'], "mode should be either '5_dfs' or '5_dfs_average'"

        # Check if the global SHAP files already exist
        global_SHAP = {}
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
                fig, axes = plt.subplots(3, 2, figsize=(30, 15), dpi=200)
                fig.suptitle(f"Global SHAP Heatmaps for {cell_line}", fontsize=16)

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
                        cbar=True,
                        vmin=vmin,
                        vmax=vmax
                    )
                    axes[i].set_title(f"Heatmap {i + 1}")
                    axes[i].set_xlabel("RBP")
                    axes[i].set_ylabel("Position")

                # Hide any unused subplots
                for j in range(len(heatmaps), len(axes)):
                    axes[j].axis("off")

                plt.tight_layout(rect=[0, 0, 1, 0.95])
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


    def calculate_SHAP_std(self):
        # Check if the SHAP_std files already exists    
        SHAP_std = {}
        for cell_line in self.cell_lines:
            if os.path.exists(self.CACHE_INFO["SHAP_std"][cell_line]): 
                SHAP_std[cell_line] = pl.read_ipc(self.CACHE_INFO["SHAP_std"][cell_line])
                logger.success(f"FROM CACHE: loaded SHAP std file for {cell_line}")

            else: 
                # Retrieve 5 SHAP tables for the cell line
                cell_line_shap = self.retrieve_5_SHAP_tables_per_cell_line(cell_line)

                # Calculate pointwise SHAP std
                output_file = self.CACHE_INFO["SHAP_std"][cell_line]
                self.calculate_pointwise_SHAP_metric_per_cell_line(cell_line_shap, 'std', output_file)
                
                SHAP_std[cell_line] = pl.read_ipc(output_file)
                logger.success(f"Saved SHAP std file for {cell_line}")

        self.SHAP_std = SHAP_std

    
    def plot_local_SHAP_std(self): 
        if not hasattr(self, "SHAP_std"): 
            self.calculate_SHAP_std()

        
