import glob, os, json, gc, pickle, gzip, tempfile, shutil, copy, sys, concurrent.futures, argparse, random

import pandas as pd, polars as pl, numpy as np
import matplotlib.pyplot as plt, matplotlib as mpl, seaborn as sns, matplotlib.gridspec as gridspec
import scipy.cluster.hierarchy as sch
import adjustText, shap, tqdm 

from dataclasses import dataclass
from IPython.display import display, Video
from loguru import logger
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from itertools import combinations
from matplotlib.legend import Legend
from sklearn.metrics import r2_score
from statannotations.Annotator import Annotator
from pathlib import Path
from matplotlib import collections as mcoll
from matplotlib.patches import Patch



@dataclass
class ShapNetworkInvestigator:
    PARAMS_DIR = "../../3_choose_dataset_and_model_parameters/output/model_reproduction/model_parameters/"
    MODEL_PICKLE_DIR = "../../4_run_final_models_and_SHAP/outputs/pickled_models/"
    CORRECTED_HAS_RBP_KD_DIR = "../../1_create_RBP_ML_input/4_create_num_peaks_ML_input/corrected_has_RBP_KD_output/"
    PRED_DIR = "../../4_run_final_models_and_SHAP/outputs/predictions/"

    SHAP_MODEL_PICKLE_DIR = "../../4_run_final_models_and_SHAP/outputs/SHAP/regular/normal/explainer_objects/"
    SHAP_DIR = "../../4_run_final_models_and_SHAP/outputs/SHAP/regular/normal/shap_values/"
    SHAP_TYPE = "regular-observational"

    FDR_THRESHOLD = 0.1
    DPSI_THRESHOLD = 0.1

    XGBOOST_BEST_MODEL_HASHES = {
        "HepG2": "fdf52464ba1145bed424d92827c559d851550a0117d96a630474c08e558ac4fd",
        "K562": "8f29591764f9e0de183047a4da90dca42b0f2847784de62970a3f20930cbe0db"
    }

    CACHE_INFO = {
            "hash_metadata": "../outputs/hash_metadata/hash_metadata.tsv",
            # "SHAP_mp4": {
            #     "K562": "../outputs/local_SHAP_distribution_video/K562_local_SHAP_distribution.mp4",
            #     "HepG2": "../outputs/local_SHAP_distribution_video/HepG2_local_SHAP_distribution.mp4",
            # },
            "SHAP_CV": {
                "All-Data": "../outputs/SHAP_cv/local_SHAP_cv.pkl.gz",
                "Unique-Binding": "../outputs/SHAP_cv/local_SHAP_cv_unique_binding.pkl.gz",
            },
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
                "Unique-Binding": {
                    "5_dfs": "../outputs/global_SHAP/5_dfs_global_SHAP_binding_unique.pkl", 
                    "5_dfs_average": "../outputs/global_SHAP/5_dfs_average_global_SHAP_binding_unique.pkl",
                },
                "All-Data": {
                    "5_dfs": "../outputs/global_SHAP/5_dfs_global_SHAP.pkl", 
                    "5_dfs_average": "../outputs/global_SHAP/5_dfs_average_global_SHAP.pkl",
                }
            },
            "specialized_global_SHAP": {
                "Bound-Only": 
                    {
                        None: 
                            {
                                "All-Data": "../outputs/specialized_global_SHAP/bound_only_global_SHAP_all_data.pkl",
                                "Unique-Binding": "../outputs/specialized_global_SHAP/bound_only_global_SHAP_unique_binding.pkl",
                            }
                    },
                "NOT-Bound-Only": 
                    {
                        None: 
                            {
                                "All-Data": "../outputs/specialized_global_SHAP/NOT_bound_only_global_SHAP_all_data.pkl",
                                "Unique-Binding": "../outputs/specialized_global_SHAP/NOT_bound_only_global_SHAP_unique_binding.pkl",
                            }
                    },
                "Signed-Local-SHAP-Mean-Bound-Only":
                    {
                        None:
                            {
                                "Unique-Binding": "../outputs/specialized_global_SHAP/signed_local_SHAP_mean_bound_only_unique_binding.pkl",
                            }
                    }, 
                "Signed-Local-SHAP-Mean-NOT-Bound-Only":
                    {   
                        None:
                            {
                                "Unique-Binding": "../outputs/specialized_global_SHAP/signed_local_SHAP_mean_NOT_bound_only_unique_binding.pkl",
                            }
                    },
            },
            "local_SHAP_mean_vs_variance": {
                "K562": "../outputs/local_SHAP_mean_vs_variance/K562_local_SHAP_mean_vs_variance.png",
                "HepG2": "../outputs/local_SHAP_mean_vs_variance/HepG2_local_SHAP_mean_vs_variance.png"
            },
            "SHAP_dispersion_per_binding_pattern": "../outputs/shap_variance_per_binding_pattern/SHAP_dispersion_per_binding_pattern.tsv", 
            "local_SHAP_mean_vs_variance_deciles": {
                "5_dfs_average": "../outputs/local_SHAP_mean_vs_variance/deciles/local_SHAP_mean_vs_variance_deciles.tsv",
                "Bound-Only": "../outputs/local_SHAP_mean_vs_variance/deciles/local_SHAP_mean_vs_variance_deciles_bound_only.tsv",
                "NOT-Bound-Only": "../outputs/local_SHAP_mean_vs_variance/deciles/local_SHAP_mean_vs_variance_deciles_NOT_bound_only.tsv",
            },
            "feature_metric_summary_table": "../outputs/feature_metric_summary_table/feature_metric_summary_table.tsv",
            "local_SHAP_percent_non_zero": {
                "All-Data": "../outputs/local_SHAP_percent_non_zero/local_SHAP_percent_non_zero_all_data.tsv",
                "Unique-Binding": "../outputs/local_SHAP_percent_non_zero/local_SHAP_percent_non_zero_unique_binding.tsv",
            },
            "local_SHAP_percent_positive_negative": {
                "NOT-Bound-Only": "../outputs/local_SHAP_percent_positive_negative/local_SHAP_percent_positive_negative_NOT_bound.pkl",
                "Bound-Only": "../outputs/local_SHAP_percent_positive_negative/local_SHAP_percent_positive_negative_bound.pkl",
            },
            "activator_repressor_behavior_score": {
                "Bound-Only": "../outputs/activator_repressor_behavior_score/activator_repressor_behavior_score_bound_only.tsv",
                "NOT-Bound-Only": "../outputs/activator_repressor_behavior_score/activator_repressor_behavior_score_NOT_bound_only.tsv",
            },
            "final_SHAP_cache": {
                "All-Data": {
                    "K562": "../outputs/FINAL_AVERAGE_SHAP_CACHE/K562_all-data.feather",
                    "HepG2": "../outputs/FINAL_AVERAGE_SHAP_CACHE/HepG2_all-data.feather"
                }, 
            },
            "per_row_num_and_percent_greater_than_cutoff": {
                "K562": "../outputs/per_row_local_shap_greater_than_cutoff/per_row_num_and_percent_greater_than_cutoff_K562.tsv.gz", 
                "HepG2": "../outputs/per_row_local_shap_greater_than_cutoff/per_row_num_and_percent_greater_than_cutoff_HepG2.tsv.gz",
            },
            "SHAP_additivity_assertions": "../outputs/SHAP_additivity_assertions/SHAP_additivity_assertions.tsv.gz",
            "position_3_4_activating_and_others_repressing": {
                "Bound Local SHAP Values": "../outputs/position_3_4_activating_others_repressing/bound_local_SHAP_values.tsv.gz", 
                "NOT Bound Local SHAP Values": "../outputs/position_3_4_activating_others_repressing/not_bound_local_SHAP_values.tsv.gz",
            }, 
            "waterfall_plot_data": "../outputs/waterfall_plot_data/waterfall_plot_data.pkl", 

        }

    non_normalized_differential_plotting_columns_info = {
        "# Diff. Events": "RBP-specific",
        "# Diff. Events + Binding (Any Pos.)": "RBP-specific",
        "# Diff. Events + Binding (Specific Pos.)": "Feature-specific"
    }

    normalized_differential_plotting_columns_info = {
        "% Diff. Events + Binding (Any Pos.)": "RBP-specific",
        "% Diff. Events + Binding (Specific Pos.)": "Feature-specific",
        "Binding Norm. Ratio Diff Events": "RBP-specific",
        "Binding Norm. Ratio Diff Events + Binding (Any Pos.)": "RBP-specific",
        "Binding Norm. Ratio Diff Events + Binding (Specific Pos.)": "Feature-specific",
    }

    binding_normalized_differential_plotting_columns_info = {
        "Binding Norm. Ratio Diff Events": "RBP-specific",
        "Binding Norm. Ratio Diff Events + Binding (Any Pos.)": "RBP-specific",
        "Binding Norm. Ratio Diff Events + Binding (Specific Pos.)": "Feature-specific",
    }

    latex_symbols = {
        "Unique-Binding": {
                "5_dfs_average": r"$\Phi_{i}$", 
                "Bound-Only": r"$\Phi_{i}^b$", 
                "NOT-Bound-Only": r"$\Phi_{i}^{nb}$",
                "local_SHAP":  r"$\varphi_{i,j}$",
                "local_SHAP_bound": r"$\varphi_{i,j}^b$",
                "local_SHAP_NOT_bound": r"$\varphi_{i,j}^{nb}$",
                "Signed-Local-SHAP-Mean-Bound-Only": r"$\overline{\varphi}_{i}^b$",
                "Signed-Local-SHAP-Mean-NOT-Bound-Only": r"$\overline{\varphi}_{i}^{nb}$",
            },
        "ElasticNet Coefficients": {
            "Absolute Value": r"$|\beta_{i}|$",
            "Signed": r"$\beta_{i}$",
        },

    }

    FIGURES = {
        "predicted_vs_actual_PSI_plot": {
            "All-Data": "../outputs/publication_figures/pred_vs_actual/predicted_vs_actual_PSI_plot_all_data.png",
            "Unique-Binding": "../outputs/publication_figures/pred_vs_actual/predicted_vs_actual_PSI_plot_unique_binding.png",
        }, 
        "global_SHAP_distribution": {
            "5_dfs_average": "../outputs/publication_figures/global_shap/global_SHAP_distribution_unique_binding.png",
            "Bound-Only": "../outputs/publication_figures/global_shap/bound_only_global_SHAP_distribution_unique_binding.png",
            "NOT-Bound-Only": "../outputs/publication_figures/global_shap/NOT_bound_only_global_SHAP_distribution_unique_binding.png",
            "All Together": "../outputs/publication_figures/global_shap/ALL_TOGETHER_global_SHAP_distribution_unique_binding.png",
            "Signed-Local-SHAP-Mean-Bound-Only": "../outputs/publication_figures/global_shap/signed_local_SHAP_mean_bound_only_distribution_unique_binding.png",
            "Signed-Local-SHAP-Mean-NOT-Bound-Only": "../outputs/publication_figures/global_shap/signed_local_SHAP_mean_NOT_bound_only_distribution_unique_binding.png",
        }, 
        "global_SHAP_heatmap": {
            "5_dfs_average": "../outputs/publication_figures/global_shap/global_SHAP_heatmap_unique_binding.png",
            "Bound-Only": "../outputs/publication_figures/global_shap/bound_only_global_SHAP_heatmap_unique_binding.png",
            "NOT-Bound-Only": "../outputs/publication_figures/global_shap/NOT_bound_only_global_SHAP_heatmap_unique_binding.png",
            "Signed-Local-SHAP-Mean-Bound-Only": "../outputs/publication_figures/global_shap/signed_local_SHAP_mean_bound_only_heatmap_unique_binding.png",
            "Signed-Local-SHAP-Mean-NOT-Bound-Only": "../outputs/publication_figures/global_shap/signed_local_SHAP_mean_NOT_bound_only_heatmap_unique_binding.png",
        },
        "global_shap_matching_features_scatter": {
            "5_dfs_average": "../outputs/publication_figures/global_shap/global_SHAP_matching_features_scatter_unique_binding.png",
            "Bound-Only": "../outputs/publication_figures/global_shap/bound_only_global_SHAP_matching_features_scatter_unique_binding.png",
            "NOT-Bound-Only": "../outputs/publication_figures/global_shap/NOT_bound_only_global_SHAP_matching_features_scatter_unique_binding.png",
            "Signed-Local-SHAP-Mean-Bound-Only": "../outputs/publication_figures/global_shap/signed_local_SHAP_mean_bound_only_matching_features_scatter_unique_binding.png",
            "Signed-Local-SHAP-Mean-NOT-Bound-Only": "../outputs/publication_figures/global_shap/signed_local_SHAP_mean_NOT_bound_only_matching_features_scatter_unique_binding.png",
        },
        "global_SHAP_alphabetical_glossary_heatmap": {
            "5_dfs_average": "../outputs/publication_figures/global_shap/global_SHAP_alphabetical_glossary_heatmap_unique_binding.png",
            "Bound-Only": "../outputs/publication_figures/global_shap/bound_only_global_SHAP_alphabetical_glossary_heatmap_unique_binding.png",
            "NOT-Bound-Only": "../outputs/publication_figures/global_shap/NOT_bound_only_global_SHAP_alphabetical_glossary_heatmap_unique_binding.png",
            "Signed-Local-SHAP-Mean-Bound-Only": "../outputs/publication_figures/global_shap/signed_local_SHAP_mean_bound_only_alphabetical_glossary_heatmap_unique_binding.png",
            "Signed-Local-SHAP-Mean-NOT-Bound-Only": "../outputs/publication_figures/global_shap/signed_local_SHAP_mean_NOT_bound_only_alphabetical_glossary_heatmap_unique_binding.png",
        },
        "position_3_4_global_shap_beta_coeff_violinplot": {
            "grouped_positions": {
                "5_dfs_average": "../outputs/publication_figures/global_shap_elasticnet_coef_violinplots/POSITION_GROUPED_position_3_4_global_shap_beta_coeff_violinplot_unique_binding.png",
                "Bound-Only": "../outputs/publication_figures/global_shap_elasticnet_coef_violinplots/POSITION_GROUPED_bound_only_position_3_4_global_shap_beta_coeff_violinplot_unique_binding.png",
                "NOT-Bound-Only": "../outputs/publication_figures/global_shap_elasticnet_coef_violinplots/POSITION_GROUPED_NOT_bound_only_position_3_4_global_shap_beta_coeff_violinplot_unique_binding.png",
            },
            "separated_positions": {
                "5_dfs_average": "../outputs/publication_figures/global_shap_elasticnet_coef_violinplots/POSITION_SEPARATED_position_3_4_global_shap_beta_coeff_violinplot_unique_binding.png",
                "Bound-Only": "../outputs/publication_figures/global_shap_elasticnet_coef_violinplots/POSITION_SEPARATED_bound_only_position_3_4_global_shap_beta_coeff_violinplot_unique_binding.png",
                "NOT-Bound-Only": "../outputs/publication_figures/global_shap_elasticnet_coef_violinplots/POSITION_SEPARATED_NOT_bound_only_position_3_4_global_shap_beta_coeff_violinplot_unique_binding.png",
            }, 
        },
        "global_shap_position_highlighting_bar_plots": {
            "Bound-Only": {
                "pos_other_highlight": "../outputs/publication_figures/global_shap_position_highlighting_bar_plots/OTHER_POS_global_shap_highlight_bar_plot.png",
                "pos_3_4_highlight": "../outputs/publication_figures/global_shap_position_highlighting_bar_plots/POS_3_4_global_shap_highlight_bar_plot.png",
            },
            "Signed-Local-SHAP-Mean-Bound-Only": {
                "pos_other_highlight": "../outputs/publication_figures/global_shap_position_highlighting_bar_plots/OTHER_POS_signed_mean_bound_local_SHAP_highlight_bar_plot.png",
                "pos_3_4_highlight": "../outputs/publication_figures/global_shap_position_highlighting_bar_plots/POS_3_4_signed_mean_bound_local_SHAP_highlight_bar_plot.png",
            }
        }, 
        "per_row_num_bound_and_percent_greater_than_cutoff": {
            "num_and_percentage_bound_local_SHAP_greater_than_0.1": "../outputs/publication_figures/per_row_local_shap_greater_than_cutoff/per_row_bound_local_SHAP_num_bound_and_percent_greater_than_0.1.png",
            "num_bound_vs_num_bound_gt_cutoff": {
                'feature': {
                    "log": "../outputs/publication_figures/per_row_local_shap_greater_than_cutoff/FEATURE_per_row_num_bound_vs_percent_greater_than_cutoff_LOG.png",
                    "linear": "../outputs/publication_figures/per_row_local_shap_greater_than_cutoff/FEATURE_per_row_num_bound_vs_percent_greater_than_cutoff_LINEAR.png",
                }, 
                'rbp': {
                    "log": "../outputs/publication_figures/per_row_local_shap_greater_than_cutoff/RBP_per_row_num_bound_vs_percent_greater_than_cutoff_LOG.png",
                    "linear": "../outputs/publication_figures/per_row_local_shap_greater_than_cutoff/RBP_per_row_num_bound_vs_percent_greater_than_cutoff_LINEAR.png",

                },
            },
        }, 
        "pct_positive_minus_negative_local_SHAP": {
            "heatmap_bound": "../outputs/publication_figures/local_SHAP_percent_positive_negative/local_SHAP_pct_pos_minus_neg_bound_heatmap.png",
            "matching_features_bound_scatter": "../outputs/publication_figures/local_SHAP_percent_positive_negative/local_SHAP_pct_pos_minus_neg_bound_matching_features_scatter.png",
        },
        "position_3_4_activating_and_others_repressing": {
            "Bound Local SHAP": "../outputs/publication_figures/middle_position_activating_others_repressing/is_position_3_4_activating_and_others_repressing_bound_local_SHAP.png",
            "NOT Bound Local SHAP": "../outputs/publication_figures/middle_position_activating_others_repressing/is_position_3_4_activating_and_others_repressing_NOT_bound_local_SHAP.png",
        },
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


    def run_data_quality_assertions(self): 
        for cell_line in self.cell_lines:

            lfs = self.get_SHAP_data_as_lazyframe(cell_line)

            for i, lf in enumerate(tqdm.tqdm(lfs, desc=f"Checking SHAP DataFrames for {cell_line}")):
                assert lf.null_count().collect().sum_horizontal().item() == 0, f"Null values found in SHAP DataFrame {i+1} for cell line {cell_line}"

                binding_cols = [col for col in lf.collect_schema().names() if col.endswith("_binding")]
                binding_values = lf.select(binding_cols).collect().to_numpy()

                assert np.isin(binding_values, [0, 1]).all(), f"One or more '_binding' columns contain values other than 0 or 1 in SHAP DataFrame {i+1} for cell line {cell_line}"

        logger.success(f"Data quality assertions passed for both cell lines.")
    

    def numpy_metric_wrapper(self, tensors, metric):

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

        return result
    

    def calculate_pointwise_SHAP_metric_per_cell_line(self, cell_line_shap=None, metric=None):
        
        assert None not in (cell_line_shap, metric), "Arguments 'cell_line_shap' and 'metric' cannot be None"
        logger.info(f"Calculating pointwise SHAP metric -- {metric}")

        # Ensure we have 5 dataframes
        assert len(cell_line_shap) == 5
        # Ensure all columns except "index" end with "_shap"
        for df in cell_line_shap:
            non_shap_cols = [col for col in df.columns if col != "index" and not col.endswith("_shap")]
            assert not non_shap_cols, f"Non-SHAP columns found: {non_shap_cols}"

        # Sort each DataFrame in cell_line_shap by the "index" column
        cell_line_shap = [df.sort("index") for df in cell_line_shap]
        # Ensure that the order of the indices matches exactly across all 5 DataFrames
        assert all(
            [(df["index"].to_list() == cell_line_shap[0]["index"].to_list()) for df in cell_line_shap]
        ), "Index order mismatch across SHAP DataFrames"
        cell_line_shap = [df.drop("index") for df in cell_line_shap]

        columns = copy.deepcopy(cell_line_shap[0].columns)
        shape = copy.deepcopy(cell_line_shap[1].shape)

        # ensure all dataframes have the same shape
        assert all(df.shape == cell_line_shap[0].shape for df in cell_line_shap), "SHAP DataFrames have inconsistent dimensions"
        # ensure all dataframes have the same columns and ordering
        assert all(df.columns == cell_line_shap[0].columns for df in cell_line_shap), "Column names are not consistent across cell_line_shap"

        if len(cell_line_shap[0].columns) < 50: 
            logger.info("Less than 50 columns in SHAP DataFrames, stacking all columns at once")
            # Convert to numpy tensors and stack
            tensors = np.stack([df.to_numpy() for df in cell_line_shap], axis=0)

            del cell_line_shap
            gc.collect()

            result = self.numpy_metric_wrapper(tensors, metric)
        
        else: 
            logger.info("More than 50 columns in SHAP DataFrames, processing in batches of 50 columns...")


            # Process columns in batches of 50, save each batch result to a temporary file with random id, then concatenate
            random_id = random.randint(10**8, 10**12)
            batch_size = 50
            num_cols = len(columns)
            temp_files = []

            for start in tqdm.tqdm(range(0, num_cols, batch_size), desc="Averaging SHAP per batch"):
                end = min(start + batch_size, num_cols)
                batch_cols = columns[start:end]

                # Stack only the relevant columns for each DataFrame
                tensors = np.stack([df.select(batch_cols).to_numpy() for df in cell_line_shap], axis=0)
                batch_result = self.numpy_metric_wrapper(tensors, metric)

                # Save batch result to a fast, uncompressed .npy file
                temp_file = f"shap_metric_tmp_{random_id}_{start}_{end}.npy"
                np.save(temp_file, batch_result)

                temp_files.append((temp_file, batch_cols))

            del batch_result
            gc.collect()

            # Concatenate all batches in the correct order
            result_arrays = []
            result_columns = []
            for temp_file, batch_cols in tqdm.tqdm(temp_files, desc="Concatenating SHAP batches"):
                arr = np.load(temp_file, allow_pickle=True)
                
                result_arrays.append(arr)
                result_columns.extend(batch_cols)
                
                os.remove(temp_file)

            # Concatenate along the last axis (columns)
            result = np.concatenate(result_arrays, axis=1)
            assert result_columns == columns, "Column names do not match the original columns"
            assert result.shape[0] == shape[0], "Resulting metric array has inconsistent number of rows"
            assert result.shape[1] == len(columns), "Resulting metric array has inconsistent number of columns"

        
        # Convert the result back to a Polars DataFrame
        result_df = pl.DataFrame(result, schema=columns)
        # Assert that the result and result_df have the same number of columns and rows as the second DataFrame in cell_line_shap
        # using second dataframe just as another double check that the first and second dataframe have the same shape
        assert result.shape == shape, "Resulting metric array has inconsistent dimensions"
        del result
        gc.collect()

        assert result_df.shape == shape, "Resulting DataFrame has inconsistent dimensions"

        logger.success(f"Calculated pointwise SHAP {metric}")
        return result_df

    
    def get_SHAP_data_as_lazyframe(self, cell_line): 

        # Filter hash metadata for rows where 'name' contains 'xgboost'
        xgboost_metadata = self.hash_metadata[self.hash_metadata['name']=="XGBRegressor"]
        group = xgboost_metadata[xgboost_metadata['cell_line'] == cell_line].sort_values(by=['hash'])
        assert len(group) == 5, f"Expected 5 SHAP files for cell line {cell_line}, but found {len(group)}"

        # Group by cell_line
        logger.info(f"Loading SHAP files for cell line {cell_line}")

        shap_dfs = []
        for hash_value in group['hash']:
            feather_file = f"{self.SHAP_DIR}/{hash_value}.feather"

            shap_dfs.append(
                pl.scan_ipc(feather_file)
            )
        
        return shap_dfs
    

    def retrieve_5_SHAP_tables_per_cell_line(self, cell_line, binding_unique=None, return_first_only=False):
        assert binding_unique in ["All-Data", "Unique-Binding"]
        assert return_first_only in [True, False], "return_first_only should be either True or False"
        
        shap_dfs =  []
        for i, df in enumerate(self.get_SHAP_data_as_lazyframe(cell_line)):
            
            schema = df.collect_schema().names()

            if binding_unique == "Unique-Binding": 
                # Subset to all columns in schema that end in "_binding" and the "index" column
                binding_columns = [col for col in schema if col.endswith("_binding")]
                # Take unique rows based on binding columns, keeping the first occurrence (lowest index)
                df = df.unique(subset=binding_columns, maintain_order=True, keep="first")

            # Subset to all columns in schema that end in "_shap" and the "index" column
            shap_columns = [col for col in schema if col.endswith("_shap")] + ["index"]
            df = df.select(shap_columns).sort('index').collect()
            
            logger.info(f"Loaded SHAP file for {cell_line}, iteration {i+1}, {binding_unique}, shape {df.shape}")

            shap_dfs.append(df)

            if i == 0 and return_first_only:
                # If we only want the first DataFrame, break after the first iteration
                break

        return shap_dfs
    

    def get_RBP_position(self, name): 
        name = name.split("_")
        assert len(name)==3, logger.error(f"Expected 3 parts in the name {name}, but got {len(name)}")
        return name[0], int(name[1])
    

    def get_slurm_job_num_cpus(self): 

        # Check if the SLURM_JOB_CPUS_PER_NODE environment variable is set
        if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
            num_cpus = int(os.environ['SLURM_NTASKS'])
            return num_cpus
        else:
            logger.warning("SLURM_JOB_CPUS_PER_NODE not set, defaulting to 1 CPU")
            return 1
    

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
        
        elif data_type == 'Final SHAP Data All Data': 
            assert hasattr(self, 'final_all_data_SHAP_data'), "final_all_data_SHAP_data attribute does not exist"
            del self.final_all_data_SHAP_data
            logger.success("Deleted final_all_data_SHAP_data attribute")
            
        elif data_type == 'Final SHAP Data Unique Binding': 
            assert hasattr(self, 'final_unique_binding_SHAP_data'), "final_unique_binding_SHAP_data attribute does not exist"
            del self.final_unique_binding_SHAP_data
            logger.success("Deleted final_unique_binding_SHAP_data attribute")

        gc.collect()


    def log_odds(self, p): 
        assert np.all((p > 0) & (p < 1)), "Input probabilities must be strictly between 0 and 1 (no exact 0 or 1 values)"
        return np.log(p / (1 - p))
    
    
    def check_no_SHAP_variance_per_binding_pattern(self): 

        # Check if the SHAP variance per binding pattern file exists
        if os.path.exists(self.CACHE_INFO["SHAP_dispersion_per_binding_pattern"]):
            # Load the SHAP variance per binding pattern file
            shap_variance_df = pd.read_csv(self.CACHE_INFO["SHAP_dispersion_per_binding_pattern"], sep="\t")
            assert shap_variance_df["num_unique_shap_rows"].nunique() == 1 and shap_variance_df["num_unique_shap_rows"].iloc[0] == 1, "num_unique_shap_rows contains values other than 1"
            logger.success("FROM CACHE: loaded SHAP variance per binding pattern stats")
            return shap_variance_df
        
        else: 

            temp_files = []
            for cell_line in self.cell_lines:
                all_results = []
                shap_lazyframes = self.get_SHAP_data_as_lazyframe(cell_line)

                for i, lf in enumerate(tqdm.tqdm(shap_lazyframes, desc=f"Processing SHAP files for {cell_line}")):
                    binding_cols = [col for col in lf.collect_schema().names() if col.endswith("_binding")]
                    shap_cols = [col for col in lf.collect_schema().names() if col.endswith("_shap")]

                    df = lf.select(binding_cols + shap_cols).collect()
                    logger.info(f"Loaded SHAP file for cell line {cell_line} with iteration {i+1} and shape {df.shape}")

                    # Group by binding pattern and count unique SHAP rows per group
                    grouped = []
                    for binding_pattern_id, group in enumerate(df.group_by(binding_cols, maintain_order=True)):
                        group_df = group[1]
                        num_unique_shap_rows = group_df.select(shap_cols).unique().height

                        # Only keep summary/statistical columns, not the actual binding/shap values
                        row = {
                            "cell_line": cell_line,
                            "model_number": i + 1,
                            "binding pattern ID": binding_pattern_id,
                            "num_unique_shap_rows": num_unique_shap_rows,
                        }
                        grouped.append(row)
                        
                    all_results.extend(grouped)

                    del df
                    gc.collect()

                # Save per-cell-line results to a file named with the cell line
                temp_file = f"shap_dispersion_{cell_line}.tsv"
                pd.DataFrame(all_results).to_csv(temp_file, sep="\t", index=False)
                temp_files.append(temp_file)

            # Concatenate all cell line files into the final output
            dfs = [pd.read_csv(f, sep="\t") for f in temp_files]
            final_df = pd.concat(dfs, ignore_index=True)
            final_df.to_csv(self.CACHE_INFO["SHAP_dispersion_per_binding_pattern"], sep="\t", index=False)

            # Clean up temporary files
            for f in temp_files:
                os.remove(f)

            return final_df


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


    def calculate_global_SHAP(self, mode=None, binding_unique=None): 
        assert mode in ['5_dfs', '5_dfs_average'], "mode should be either '5_dfs' or '5_dfs_average'"
        assert binding_unique in ["All-Data", "Unique-Binding"], "binding_unique should be either True or False"

        output_file = self.CACHE_INFO["global_SHAP"][binding_unique][mode]
        # Check if the output file exists
        if os.path.exists(output_file):
            with open(output_file, 'rb') as f:
                global_SHAP = pickle.load(f)
            logger.success(f"FROM CACHE: loaded global SHAP file for mode {mode} and {binding_unique}")
            return global_SHAP
        
        else:
            logger.info(f"Global SHAP file for mode {mode} and {binding_unique} does not exist. Calculating...")
            global_heatmaps = {}

            if mode == '5_dfs':
                dfs_global_SHAP = {
                    cell_line: self.retrieve_5_SHAP_tables_per_cell_line(cell_line, binding_unique=binding_unique) for cell_line in self.cell_lines
                }

                for cell_line, dfs in dfs_global_SHAP.items():
                    # Drop the 'index' column and calculate the absolute value average of each column
                    averaged_df = [df.sort('index').drop('index').select(pl.all().abs().mean()) for df in dfs]
                    # Convert each averaged DataFrame to a 2D heatmap
                    heatmaps = [self.convert_RBP_position_to_2d_heatmap(df) for df in averaged_df]
                    # Store the heatmaps in the global dictionary
                    global_heatmaps[cell_line] = heatmaps

            elif mode == '5_dfs_average':

                lazy_final_SHAP = self.load_final_SHAP_data(underlying_data=binding_unique, as_lazyframe=True)
                for cell_line in lazy_final_SHAP: 
                    shap_cols = [col for col in lazy_final_SHAP[cell_line].collect_schema().names() if col.endswith("_shap")]
                    # Calculate the average of the absolute values of the SHAP columns
                    average_df = lazy_final_SHAP[cell_line].select(
                        [pl.col(col).abs().mean() for col in shap_cols]
                    ).collect()

                    # Convert the averaged DataFrame to a 2D heatmap
                    heatmap = self.convert_RBP_position_to_2d_heatmap(average_df)
                    # Store the heatmap in the global dictionary
                    global_heatmaps[cell_line] = heatmap
            
            assert len(global_heatmaps.keys()) == len(self.cell_lines), "Not all cell lines have been processed"
            # Save the global heatmaps as a single pickle file
            with open(output_file, 'wb') as f:
                pickle.dump(global_heatmaps, f)

            logger.success(f"Saved global SHAP file for mode {mode} and {binding_unique} to {output_file}")


    def plot_global_SHAP(self, mode=None, binding_unique=None):
        VALID_MODES = ['5_dfs', '5_dfs_average', 'Bound-Only', 'NOT-Bound-Only', 'NOT-Bound-Only-CTRL', 'NOT-Bound-Only-RBP_KD', 'NOT-Bound-Only-RBP_KD_at_position', "Signed-Local-SHAP-Mean-Bound-Only", "Signed-Local-SHAP-Mean-NOT-Bound-Only"]
        assert mode in VALID_MODES, f"mode should be one of {VALID_MODES}"
        assert binding_unique in ["All-Data", "Unique-Binding"], "binding_unique should be either 'All-Data' or 'Unique-Binding'"
        
        if mode == '5_dfs' or mode == '5_dfs_average':
            global_SHAP = self.calculate_global_SHAP(mode, binding_unique)
        else: 

            if mode in ['Bound-Only', 'NOT-Bound-Only', 'Signed-Local-SHAP-Mean-Bound-Only', 'Signed-Local-SHAP-Mean-NOT-Bound-Only']:
                global_SHAP = self.calculate_specialized_global_SHAP(mode=mode, condition=None, underlying_data = binding_unique)
            elif mode in ['NOT-Bound-Only-CTRL', 'NOT-Bound-Only-RBP_KD', 'NOT-Bound-Only-RBP_KD_at_position']:
                global_SHAP = self.calculate_specialized_global_SHAP(mode="NOT-Bound-Only", condition=mode.split('-')[-1], underlying_data = binding_unique)

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

        else: 

            # Calculate the combined range of all heatmaps to define consistent bins
            all_values = np.concatenate([heatmap.to_numpy().flatten() for heatmap in global_SHAP.values()])
            
            if "Bound-Only" in mode: 
                all_values = all_values[~np.isnan(all_values)]  # Remove NaN values
            else: 
                assert not np.isnan(all_values).any(), "NaN values found in all_values"   
                
            if mode == '5_dfs_average':
                prefix = binding_unique.replace('-', ' ')
            elif mode in ['Bound-Only', 'NOT-Bound-Only', 'Signed-Local-SHAP-Mean-Bound-Only', 'Signed-Local-SHAP-Mean-NOT-Bound-Only']:
                prefix = mode.replace('-', ' ')
            elif mode in ['NOT-Bound-Only-CTRL', 'NOT-Bound-Only-RBP_KD', 'NOT-Bound-Only-RBP_KD_at_position']:
                prefix = f"{' '.join(mode.split('-')[:-1])} ({mode.split('-')[-1]})"
            
            # Create a figure with 2 columns: left for histograms, right for boxplots
            fig, axes = plt.subplots(len(global_SHAP), 2, figsize=(11, 7), dpi=200, sharex=True, sharey=False)
            bins = np.linspace(all_values.min(), all_values.max(), 21)  # Define 20 equal-width bins

            for row_idx, (cell_line, heatmap) in enumerate(global_SHAP.items()):
                # Flatten the heatmap values into a single array
                global_shap_values = heatmap.to_numpy().flatten()
                global_shap_values = global_shap_values[~np.isnan(global_shap_values)]
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

            plt.suptitle(f"{prefix}: Global SHAP per Cell Line from Avg. 5 Models' Local SHAP", fontsize=16, y=0.98)
            fig.supxlabel("Global SHAP Value", fontsize=16)
            fig.supylabel("Percentage", fontsize=16)
            plt.tight_layout()
            plt.show()

            # Prepare data for violinplot: melt global_SHAP into long format
            violin_data = []
            for cell_line, heatmap in global_SHAP.items():
                for pos in heatmap.index:
                    for rbp in heatmap.columns:
                        value = heatmap.at[pos, rbp]
                        violin_data.append({"Cell Line": cell_line, "Global SHAP": value})
            violin_df = pd.DataFrame(violin_data)

            plt.figure(figsize=(5,3), dpi=300)
            ax = plt.gca()

            # Violinplot with boxplot inside, grouped by cell line
            # Use light orange for HepG2 and light blue for K562
            palette = {"HepG2": "#FFD580", "K562": "#ADD8E6"}
            sns.violinplot(
                data=violin_df,
                x="Cell Line",
                y="Global SHAP",
                inner=None,
                palette=palette,
                cut=0,
                linewidth=1,
                edgecolor="black",
                density_norm="width",
                ax=ax
            )
            
            sns.boxplot(
                data=violin_df,
                x="Cell Line",
                y="Global SHAP",
                width=0.2,
                showcaps=True,
                showfliers=True,
                boxprops={"facecolor": "none", "edgecolor": "black", "zorder": 2},
                meanline=True,
                showmeans=True,
                meanprops={"color": "red", "linestyle": "--", "linewidth": 1.5},
                flierprops={"marker": "o", "markersize": 2, "markerfacecolor": "gray", "alpha": 0.5},
                ax=ax
            )

            # Set y-axis limit to be 0 to 20% larger than the current max
            _, ymax = ax.get_ylim()
            ax.set_ylim(-0.02, ymax * 1.25)

            # Draw dashed line for the mean per cell line
            for i, cell_line in enumerate(sorted(violin_df["Cell Line"].unique())):
                vals = violin_df[violin_df["Cell Line"] == cell_line]["Global SHAP"].dropna()
                # mean_val = vals.mean()
                # ax.hlines(mean_val, i - 0.3, i + 0.3, colors="red", linestyles="--", linewidth=2, zorder=3)
                n_points = len(vals)
                median_val = np.median(vals)
                pct_zero = (vals == 0).mean() * 100
                # Place annotation inside the plot area, just below the top y-limit
                ax.text(
                    i, ax.get_ylim()[1] - 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    f"# Values: {n_points}\nMedian: {median_val:.3g}\n% Zero: {pct_zero:.1f}",
                    ha="center", va="top", fontsize=7, color="black"
                )

            ax.set_title(f"{binding_unique} - {mode}: Global SHAP Value Distribution per Cell Line", fontsize=6, y=1.05)
            ax.set_xlabel("Cell Line", fontsize=10)
            ax.set_ylabel(self.latex_symbols[binding_unique][mode], fontsize=14)
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)

            plt.tight_layout()
            plt.savefig(self.FIGURES["global_SHAP_distribution"][mode], dpi=600, bbox_inches='tight')
            plt.show()

            for iteration, log_scale in enumerate([False, True]):
                fig, axes = plt.subplots(2, 1, figsize=(35, 17), dpi=200, sharey=True)

                for ax, (cell_line, heatmap) in zip(axes, global_SHAP.items()):
                    # Perform hierarchical clustering on the columns
                    linkage = sch.linkage(heatmap.T.fillna(0), method="ward")
                    dendrogram = sch.dendrogram(linkage, no_plot=True)
                    ordered_columns = [heatmap.columns[i] for i in dendrogram["leaves"]]
                    # Reorder the heatmap columns based on the clustering
                    ordered_heatmap = heatmap[ordered_columns]

                    # Set norm for log scale if needed
                    if log_scale:
                        # norm = LogNorm(vmin=max(heatmap.min().min(), 1e-6), vmax=heatmap.max().max())
                        norm=LogNorm()
                    else:
                        norm = None

                    heatmap_kwargs = dict(
                        data=ordered_heatmap,
                        ax=ax,
                        cmap= "seismic" if mode.startswith("Signed-Local-SHAP-Mean") else "Blues",
                        cbar=True,
                        linewidths=0.01,  # Add black border around each cell
                        linecolor="gray",
                        cbar_kws={"shrink": 1, "aspect": 20, "pad": 0.02},  # Adjust colorbar position and size
                        vmin=None,
                        center= 0 if mode.startswith("Signed-Local-SHAP-Mean") else None,
                        norm=norm,
                        annot=True,
                        fmt=".3f",  # Default annotation format
                        annot_kws={"size": 14, "rotation": 90},
                    )
                    
                    set_bad_color=False
                    if mode in ["Bound-Only", "NOT-Bound-Only-RBP_KD", "NOT-Bound-Only-RBP_KD_at_position", "Signed-Local-SHAP-Mean-Bound-Only"]: 
                        heatmap_kwargs["mask"] = ordered_heatmap.isnull()
                        set_bad_color = True

                    sns.heatmap(**heatmap_kwargs)

                    if set_bad_color:
                        ax.set_facecolor("black")

                    cbar = ax.collections[0].colorbar
                    cbar.ax.tick_params(labelsize=26)  # Make colorbar tick labels larger
                    ax.set_title(f"{cell_line}", fontsize=35, pad=15)
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.tick_params(axis='y', labelsize=30)  # Make y-axis tick labels larger
                    ax.tick_params(axis='x', labelsize=13)

                # Add a caption for the log scale iteration
                caption = ""
                if log_scale:
                    caption = "\nNOTE 2: colorbar is log-scaled and only shows values > 0."

                fig.suptitle(
                    f"{prefix}: Global SHAP w/ Ward Hierarchical Clustering Order\n"
                    f"NOTE: after averaging all local SHAP values across 5 models per cell line{caption}",
                    fontsize=40, y=1.01, x=0.45
                )
                fig.supxlabel("RBP", fontsize=40, x=0.43)
                fig.supylabel("Position", fontsize=40, x=-0.005)
                plt.tight_layout()

                if not log_scale: 
                    plt.savefig(self.FIGURES["global_SHAP_heatmap"][mode], dpi=600, bbox_inches='tight')

                plt.show()

            # Plot dendrograms for hierarchical clustering of rows (positions) using Ward linkage, all in one figure
            n_cell_lines = len(global_SHAP)
            fig, axes = plt.subplots(1, n_cell_lines, figsize=(4* n_cell_lines, 3), dpi=300, squeeze=False)
            for idx, (cell_line, heatmap) in enumerate(global_SHAP.items()):
                # Perform hierarchical clustering on the rows (positions)
                linkage_rows = sch.linkage(heatmap.fillna(0), method="ward")
                ax = axes[0, idx]
                sch.dendrogram(linkage_rows, labels=heatmap.index, orientation="top", color_threshold=None, ax=ax)
                ax.set_title(f"{cell_line}", fontsize=14)
                ax.set_xlabel("")
                ax.set_ylabel("")
            
            plt.suptitle(f"{prefix} {binding_unique}: Ward Clustering of Positions\nNOTE 1: Null values replaced with 0 for clustering", fontsize=12, y=1.1) 
            fig.supxlabel("Position", fontsize=12)
            fig.supylabel("Distance", fontsize=12)
            
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
            if "Bound-Only" in mode: 
                combined_df = combined_df.dropna(axis = 0, how = 'any')
            # Identify the top 5 features with the highest global SHAP values in HepG2 and K562

            if mode == "Bound-Only" or mode.startswith("Signed-Local-SHAP-Mean-"): 

                if mode == "Signed-Local-SHAP-Mean-Bound-Only": 
                    n_largest_smallest = 20
                elif mode == "Signed-Local-SHAP-Mean-NOT-Bound-Only": 
                    n_largest_smallest = 10
                elif mode == "Bound-Only": 
                    n_largest = 25

                fontsize= 3

            else: 
                n_largest = 10
                fontsize= 5

            if mode.startswith("Signed-Local-SHAP-Mean-"): 

                highest_hepg2_features = combined_df.nlargest(n_largest_smallest, "HepG2")
                highest_k562_features = combined_df.nlargest(n_largest_smallest, "K562")

                lowest_hepg2_features = combined_df.nsmallest(n_largest_smallest, "HepG2")
                lowest_k562_features = combined_df.nsmallest(n_largest_smallest, "K562")
                top_features = pd.concat([highest_k562_features, highest_hepg2_features, lowest_hepg2_features, lowest_k562_features]).drop_duplicates()

            else: 
                top_hepg2_features = combined_df.nlargest(n_largest, "HepG2")
                top_k562_features = combined_df.nlargest(n_largest, "K562")
                top_features = pd.concat([top_hepg2_features, top_k562_features]).drop_duplicates()

            fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
            plot_types = [("Linear", None, None), ("Log-Log", "log", "log")]

            for i, (label, xscale, yscale) in enumerate(plot_types):
                ax = axes[i]
                
                if mode in ["Bound-Only", "NOT-Bound-Only-RBP_KD", "NOT-Bound-Only-RBP_KD_at_position", "Signed-Local-SHAP-Mean-Bound-Only"]:
                    combined_df = combined_df.dropna(subset=["HepG2", "K562"])

                x = combined_df["HepG2"]
                y = combined_df["K562"]

                # For log-log, filter out non-positive values
                if xscale == "log" and yscale == "log":
                    mask = (x > 0) & (y > 0)
                    x = x[mask]
                    y = y[mask]
                    features = combined_df["Feature"][mask]
                else:
                    features = combined_df["Feature"]

                # Calculate correlations
                pearson_corr, _ = pearsonr(x, y)
                spearman_corr, _ = spearmanr(x, y)

                sns.scatterplot(
                    x=x,
                    y=y,
                    alpha=0.7,
                    edgecolor="black",
                    color="deepskyblue",
                    s=20,
                    ax=ax
                )

                # Add y=x line
                min_val = min(x.min(), y.min())
                max_val = max(x.max(), y.max())
                ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1, label="y=x")
                ax.legend(loc = "upper center", fontsize=10)

                if xscale:
                    ax.set_xscale(xscale)
                if yscale:
                    ax.set_yscale(yscale)

                ax.set_title(f"{label} Scale", fontsize=14)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.text(
                    0.02, 0.95,
                    f"Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}\nPoints: {len(x)}",
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left'
                )

                # Annotate top features (only on linear plot for clarity)
                if i == 0:
                    texts = []
                    for _, row in top_features.iterrows():
                        texts.append(
                            ax.text(
                                row["HepG2"],
                                row["K562"],
                                row["Feature"],
                                fontsize=fontsize,
                                color="#A34C00",
                                alpha=0.7
                            )
                        )
                    
                    adjustText.adjust_text(
                        texts,
                        ax=ax,
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                        force_points=2.0,
                        force_text=2.0,
                        expand_points=(1.2, 1.2),
                        expand_text=(1.2, 1.2),
                        only_move={'points': 'y', 'text': 'y'}
                    )

            plt.suptitle(f"{prefix}: Global SHAP Values for Matching Features Across Cell Lines\nNOTE: log scale only includes values > 0", fontsize=14, y=1.02)
            fig.supxlabel(f"HepG2 {self.latex_symbols[binding_unique][mode]}", fontsize=14) 
            fig.supylabel(f"K562 {self.latex_symbols[binding_unique][mode]}", fontsize=14)

            plt.tight_layout()
            plt.savefig(self.FIGURES["global_shap_matching_features_scatter"][mode], dpi=600, bbox_inches='tight')
            plt.show()

            # Alphabetical Glossary Global SHAP Heatmap (transposed: RBPs as rows, positions as columns)
            hepg2_df = global_SHAP["HepG2"].copy()
            k562_df = global_SHAP["K562"].copy()

            # Get union of RBPs profiled in both cell lines and sort alphabetically
            hepg2_rbps = set(hepg2_df.columns)
            k562_rbps = set(k562_df.columns)
            union_rbps = sorted(hepg2_rbps.union(k562_rbps))

            positions = sorted(set(hepg2_df.index).union(set(k562_df.index)))

            # Prepare heatmap data and null type for each cell line using iteration
            heatmaps = {}
            null_types = {}
            dfs = {"HepG2": hepg2_df, "K562": k562_df}

            for cell_line, df in dfs.items():
                # Transpose: RBPs as rows, positions as columns
                heatmap = pd.DataFrame(np.nan, index=union_rbps, columns=positions)
                null_type = pd.DataFrame("not-profiled", index=union_rbps, columns=positions)

                for pos in df.index:
                    for rbp in df.columns:
                        val = df.at[pos, rbp]
                        heatmap.at[rbp, pos] = val
                        null_type.at[rbp, pos] = "not-bound" if pd.isnull(val) else "value"

                heatmaps[cell_line] = heatmap
                null_types[cell_line] = null_type

            # Predefine color scale for both cell lines
            global_min = min(h.min().min() for h in heatmaps.values())
            global_max = max(h.max().max() for h in heatmaps.values())

            # Plot both cell lines in one figure with a single shared colorbar axis (side by side)
            fig, axes = plt.subplots(
                1, 2, figsize=(18, 45), dpi=300, sharex=True, sharey=True,
                gridspec_kw={'wspace': 0.1}  # Increase space between columns
            )
            # Make colorbar wider and move further right
            cbar_ax = fig.add_axes([0.93, 0.4, 0.025, 0.4])  # Wider colorbar
            # Move null type legend further right to avoid overlap
            legend_ax = fig.add_axes([1.05, 0.23, 0.04, 0.2])  # Further right

            for idx, cell_line in enumerate(["HepG2", "K562"]):
                heatmap = heatmaps[cell_line]
                null_type = null_types[cell_line]

                mask = heatmap.isnull()
                annot = heatmap.round(3)

                cmap = "seismic" if mode.startswith("Signed-Local-SHAP-Mean") else "Blues"
                center = 0 if mode.startswith("Signed-Local-SHAP-Mean") else None

                sns.heatmap(
                    heatmap,
                    ax=axes[idx],
                    cmap=cmap,
                    center=center,
                    linewidths=.7,
                    linecolor="black",
                    annot=annot,
                    fmt=".3f",
                    annot_kws={"size": 14, "rotation": 0},  # Rotate annotation text to horizontal
                    mask=mask,
                    cbar=(idx == 0),  # Only add colorbar for the first plot
                    cbar_ax=(cbar_ax if idx == 0 else None),
                    cbar_kws={"shrink": 1, "aspect": 30, "pad": 0.02},  # Wider colorbar
                    vmin=global_min,
                    vmax=global_max,
                )

                # Overlay null markers
                for i, rbp in enumerate(union_rbps):
                    for j, pos in enumerate(positions):
                        if mask.at[rbp, pos]:
                            null_kind = null_type.at[rbp, pos]
                            color = "#FF991C"
                            if null_kind == "not-bound":
                                # Circle marker for "not bound"
                                axes[idx].scatter(j + 0.5, i + 0.5, marker="D", s=60, color=color, edgecolor="black", linewidths=0.75, zorder=10)
                            elif null_kind == "not-profiled":
                                # Lowercase x marker
                                axes[idx].scatter(j + 0.5, i + 0.5, marker="X", s=60, color=color, edgecolor="black", linewidths=0.75, zorder=10)

                axes[idx].set_title(f"{cell_line}", fontsize=50, pad=20)  # Smaller title font
                axes[idx].set_xlabel("")
                axes[idx].set_ylabel("")
                axes[idx].tick_params(axis='y', labelsize=14)  # 50% smaller than 36
                axes[idx].tick_params(axis='x', labelsize=45)  # Larger x-axis tick labels

            cbar_ax.set_title(self.latex_symbols[binding_unique][mode], fontsize=50, pad=20, loc="left")
            cbar_ax.tick_params(labelsize=40)

            fig.supxlabel("Position", fontsize=60, x=.51, y=0.07)
            fig.supylabel("RBP\n(Alphabetically Sorted)", fontsize=50, x=0.01, ha="center")
            plt.suptitle(
                f"{prefix}: {self.latex_symbols[binding_unique][mode]} Alphabetical Glossary Heatmap\nNOTE: RBPs are union of both cell lines, sorted alphabetically",
                fontsize=18, y=0.92, x=0.51  # Smaller suptitle font
            )

            # Add legend for null types before tight_layout
            legend_ax.axis("off")
            
            legend_ax.scatter([], [], marker="X", s=10, color="#FF991C", edgecolor="black", linewidths=1, label="Not Profiled")
            if mode in ["Bound-Only", "Signed-Local-SHAP-Mean-Bound-Only"]:
                legend_ax.scatter([], [], marker="D", s=10, color="#FF991C", edgecolor="black", linewidths=1, label="Not Bound")
            
            legend_ax = legend_ax.legend(
                title="Null Type" if mode in ["Bound-Only", "Signed-Local-SHAP-Mean-Bound-Only"] else "",
                loc="center",
                fontsize=40,
                title_fontsize=50,
                frameon=True,
                markerscale=12,
                edgecolor="black",
            )
            legend_ax.get_frame().set_linewidth(2)  # Thicker border for the legend box

            plt.tight_layout(rect=[0, 0, 0.89, 1])  # Call after legend, leave space for legend/colorbar

            plt.savefig(self.FIGURES["global_SHAP_alphabetical_glossary_heatmap"][mode], dpi=600, bbox_inches='tight')
            plt.show()

    
    def plot_global_SHAP_mean_vs_variance(self, mode=None, binding_unique=None):
        assert mode =="5_dfs", "mode should be '5_dfs'"
        assert binding_unique in ["All-Data", "Unique-Binding"], "binding_unique should be either 'All-Data' or 'Unique-Binding'"
        
        global_SHAP = self.calculate_global_SHAP(mode, binding_unique)

        if mode == '5_dfs':

            return_results = {}
            # First, compute the mean and variance tables for each cell line
            for cell_line, heatmaps in global_SHAP.items():
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

            # Plot 2 rows (scales) x N cell lines (columns)
            n_cell_lines = len(return_results)
            cell_lines = list(return_results.keys())
            scales = [("linear", None, None), ("log-log", "log", "log")]

            fig, axes = plt.subplots(len(scales), n_cell_lines, figsize=(12, 9), dpi=300, sharex="row", sharey="row")
            for col_idx, cell_line in enumerate(cell_lines):
                result_table = return_results[cell_line]

                for row_idx, (label, xscale, yscale) in enumerate(scales):
                    ax = axes[row_idx, col_idx]
                    plot_data = result_table.copy()

                    # Apply log scale if needed for correlation calculation
                    x = plot_data["Mean"]
                    y = plot_data["Variance"]

                    if xscale == "log" and yscale == "log":
                        # Only keep points where both mean and variance are > 0
                        mask = (x > 0) & (y > 0)
                        x = np.log10(x[mask])
                        y = np.log10(y[mask])
                        plot_data = plot_data[mask]

                    # Assert all values are finite real numbers after log transform
                    assert np.isfinite(x).all(), "Non-finite values found in log10(mean)"
                    assert np.isfinite(y).all(), "Non-finite values found in log10(variance)"

                    num_points = len(plot_data)
                    top5 = plot_data.nlargest(10, "Variance")

                    pearson_corr, _ = pearsonr(x, y)
                    spearman_corr, _ = spearmanr(x, y)

                    sns.scatterplot(data=plot_data, x="Mean", y="Variance", alpha=0.7, edgecolor="black", color="lightskyblue", ax=ax)
                    
                    if xscale:
                        ax.set_xscale(xscale)
                    if yscale:
                        ax.set_yscale(yscale)

                    ax.set_title(f"{cell_line} ({label.replace('-', ' ').title()})", fontsize=16)
                    ax.set_xlabel("")
                    ax.set_ylabel("")

                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}"))

                    ax.text(
                        0.03, 0.9,
                        f"Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}\nPoints: {num_points}",
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment='center',
                        horizontalalignment='left'
                    )

                    # only annotate top row
                    if row_idx == 0:
                        for _, row in top5.iterrows():
                            ax.text(
                                row["Mean"]-0.001, row["Variance"]+0.00000001,
                                row["RBP_Position"],
                                fontsize=6, color="black", alpha=0.8
                            )

            plt.suptitle(
                "Mean vs Variance of Global SHAP per Feature Across 5 Models\nNOTE 1: top row is linear scale, bottom row is log-log scale\nNOTE 2: log-log scale only shows points with both mean and variance > 0",
                fontsize=16, 
                y=0.99
            )
            plt.tight_layout()

            plt.show()
            plt.close()

            return return_results


    def plot_global_SHAP_coefficient_of_variation(self, mode=None, binding_unique=None):
        assert mode == '5_dfs', "mode should be '5_dfs'"
        assert binding_unique in ["All-Data", "Unique-Binding"], "binding_unique should be either 'All-Data' or 'Unique-Binding'"

        global_SHAP = self.calculate_global_SHAP(mode=mode, binding_unique=binding_unique)

        # Assert that all 5 heatmaps per cell line have the same shape, columns, and index order
        for cell_line, heatmaps in global_SHAP.items():
            assert all(h.shape == heatmaps[0].shape for h in heatmaps), f"Heatmaps for {cell_line} have inconsistent shapes"
            assert all(h.columns.equals(heatmaps[0].columns) for h in heatmaps), f"Column order mismatch for {cell_line}"
            assert all(h.index.equals(heatmaps[0].index) for h in heatmaps), f"Index order mismatch for {cell_line}"
    
        # Compute coefficient of variation and mean for each cell line
        cv_results = []
        for cell_line, heatmaps in global_SHAP.items():
            # Stack into 3D array: (5, n_rows, n_cols)
            stacked = np.stack([h.values for h in heatmaps], axis=0)
            mean = np.mean(stacked, axis=0)
            std = np.std(stacked, axis=0)
            cv = std / mean

            # Flatten and collect results
            mean_flat = mean.flatten()
            cv_flat = cv.flatten()
            for m, v in zip(mean_flat, cv_flat):
                if np.isfinite(m) and np.isfinite(v):
                    cv_results.append({"Cell Line": cell_line, "Mean": m, "CV": v})

        cv_df = pd.DataFrame(cv_results)
        cv_df = cv_df.sort_values("Cell Line")

        plt.figure(figsize=(6, 4), dpi=200)
        sns.scatterplot(
            data=cv_df,
            x="Mean",
            y="CV",
            hue="Cell Line",
            palette="Set2",
            alpha=0.5,
            edgecolor="black",
            s=1
        )
        plt.title("Mean vs Coefficient of Variation of Global SHAP Across 5 Models per Cell Line")
        plt.ylabel("Coefficient of Variation")
        plt.xlabel("Mean Global SHAP")
        plt.legend(title="Cell Line")
        plt.tight_layout()
        plt.show()

        # Swarmplot of coefficient of variation per cell line
        plt.figure(figsize=(6, 4), dpi=200)
        sns.swarmplot(
            data=cv_df,
            x="Cell Line",
            y="CV",
            palette="Set2",
            alpha=0.7,
            edgecolor="black",
            size=2
        )
        plt.title("Coefficient of Variation of Global SHAP Across 5 Models per Cell Line", fontsize=12)
        plt.ylabel("Coefficient of Variation")
        plt.xlabel("Cell Line")

        # Annotate number of points above each cell line
        for idx, cell_line in enumerate(cv_df["Cell Line"].unique()):
            n_points = (cv_df["Cell Line"] == cell_line).sum()
            plt.text(idx, cv_df[cv_df["Cell Line"] == cell_line]["CV"].max() + 0.02, f"n={n_points}", 
                    ha="center", va="bottom", fontsize=10, color="black")

        plt.tight_layout()
        plt.show()


    def plot_global_SHAP_between_unique_binding_and_all_data(self): 
        # Get global SHAP for both modes
        global_shap_all = self.calculate_global_SHAP(mode="5_dfs_average", binding_unique="All-Data")
        global_shap_unique = self.calculate_global_SHAP(mode="5_dfs_average", binding_unique="Unique-Binding")

        for log_scale in [False, True]:
            fig, axes = plt.subplots(2, 1, figsize=(35, 18), dpi=300)

            for i, cell_line in enumerate(self.cell_lines):
                # Get heatmaps for both modes
                df_all = global_shap_all[cell_line]
                df_unique = global_shap_unique[cell_line]

                # Ensure both DataFrames have the same columns and index after sorting
                df_all = df_all.sort_index().sort_index(axis=1)
                df_unique = df_unique.sort_index().sort_index(axis=1)
                assert (df_all.columns.equals(df_unique.columns) and df_all.index.equals(df_unique.index)), "df_all and df_unique must have the same columns and index after sorting"

                # Subtract unique - all
                diff = df_unique - df_all
                assert diff.shape == df_all.shape == df_unique.shape, "Difference DataFrame must have the same shape as original DataFrames"
                assert diff.isnull().sum().sum() == 0, "Difference DataFrame contains null values"

                # Cluster columns (RBPs) using Ward, keep original row (position) order
                col_linkage = sch.linkage(diff.T, method="ward")
                col_dendro = sch.dendrogram(col_linkage, no_plot=True)

                # Reorder only columns
                ordered_cols = [diff.columns[i] for i in col_dendro["leaves"]]
                diff_ordered = diff.loc[diff.index, ordered_cols]

                # Plot
                if log_scale:
                    norm = LogNorm()
                else:
                    norm = None
                    
                sns.heatmap(
                    diff_ordered,
                    ax=axes[i],
                    cmap="bwr",
                    center=0,
                    linewidths=0.5,
                    linecolor="gray",
                    annot=True,
                    fmt=".2f",
                    annot_kws={"size": 14, "rotation": 90},
                    cbar_kws={"shrink": 0.9, "aspect": 15, "pad": 0.01},
                    norm=norm,
                )

                # Make colorbar tick labels larger
                cbar = axes[i].collections[0].colorbar
                cbar.ax.tick_params(labelsize=20)

                axes[i].set_title(f"{cell_line}", fontsize=26)
                axes[i].set_xlabel("")
                axes[i].set_ylabel("")
                axes[i].tick_params(axis='y', labelsize=26)
                axes[i].tick_params(axis='x', labelsize=12)

            note = ""
            if log_scale:
                note = "\nNOTE 2: colorbar is log-scaled and only shows values > 0."
            plt.suptitle("Global SHAP Difference: 'Unique Binding' - 'All Data'\nNOTE: each heatmap clustered with Ward" + note, fontsize=30, y=1.02)
            fig.supxlabel("RBP", fontsize=40)
            fig.supylabel("Position", fontsize=40, x=-0.01)

            plt.tight_layout()
            plt.show()


    def calculate_SHAP_CV(self, binding_unique=None):
        assert binding_unique == "Unique-Binding", "binding_unique should be 'Unique-Binding'"

        output_file = self.CACHE_INFO["SHAP_CV"]
        # Check if the output file exists
        if os.path.exists(output_file):
            with gzip.open(output_file, 'rb') as f:
                SHAP_cv = pickle.load(f)
            
            # Convert each DataFrame in SHAP_cv from pandas to polars
            self.SHAP_cv = {cell_line: pl.from_pandas(df) for cell_line, df in SHAP_cv.items()}
            logger.success(f"FROM CACHE: loaded SHAP CV file for {binding_unique}")  

        else:
            logger.info(f"SHAP CV file does not exist. Calculating for {binding_unique}...")

            # Initialize an empty dictionary to store SHAP CV results
            SHAP_CV = {}
            # Calculate the coefficient of variation for each cell line
            for cell_line in self.cell_lines:
                logger.info(f"Calculating SHAP CV for cell line {cell_line}")

                # Retrieve the 5 SHAP DataFrames for the cell line
                shap_dfs = self.retrieve_5_SHAP_tables_per_cell_line(cell_line, binding_unique=binding_unique)
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

    
    def plot_SHAP_CV(self, binding_unique=None):
        assert binding_unique == "Unique-Binding", "binding_unique should be 'Unique-Binding'"
        
        if not hasattr(self, 'SHAP_cv'):
            self.calculate_SHAP_CV(binding_unique = binding_unique)

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
                shap_dfs = self.retrieve_5_SHAP_tables_per_cell_line(cell_line, binding_unique="All-Data")
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
                shap_dfs = self.retrieve_5_SHAP_tables_per_cell_line(cell_line, binding_unique="Unique-Binding")

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


    def calculate_local_SHAP_mean_vs_variance_deciles(self, mode=None):
        assert mode in ["Bound-Only", "NOT-Bound-Only"], "mode should be either 'Bound-Only' or 'NOT-Bound-Only'"

        # Check if the local SHAP mean vs variance deciles TSV file exists
        output_file = self.CACHE_INFO["local_SHAP_mean_vs_variance_deciles"][mode]
        if os.path.exists(output_file):
            logger.success(f"FROM CACHE: Local SHAP mean vs variance deciles table already exists for mode: {mode}")
            decile_df = pd.read_csv(output_file, sep="\t")
            for cell_line in self.cell_lines:
                logger.success(f"Loaded mean vs variance deciles table for {cell_line} from {output_file}")
                display(decile_df[decile_df["Cell Line"] == cell_line])
        else:
            logger.info(f"Local SHAP mean vs variance deciles not calculated. Computing for {mode} ...")

            all_deciles = []
            local_shap = self.get_local_SHAP_based_on_binding_and_covariates(
                binding_value = 1 if mode == "Bound-Only" else 0,
                binding_pattern_type = "Unique-Binding"
            )

            # Collect all local SHAP values per cell line
            cell_line_shap_values = {cell_line: [] for cell_line in self.cell_lines}
            for cell_line, shap_col, series in local_shap:
                # Concatenate all values for this cell line
                cell_line_shap_values[cell_line].append(series.abs().to_numpy())

            # For each cell line, concatenate all arrays and compute deciles
            for cell_line in self.cell_lines:
                all_values = np.concatenate(cell_line_shap_values[cell_line])
                percentiles = np.arange(0, 101, 11)
                decile_values = np.percentile(all_values, percentiles)
                for perc, val in zip(percentiles, decile_values):
                    all_deciles.append({
                        "Cell Line": cell_line,
                        "Percentile": perc,
                        "Value": val
                    })

            # Save as DataFrame and TSV
            decile_df = pd.DataFrame(all_deciles)
            decile_df.to_csv(output_file, sep="\t", index=False)

            logger.success(f"Saved local SHAP mean vs variance deciles for mode {mode} to {output_file}")


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

    
    def check_position_3_and_4_significance_for_global_SHAP_and_elasticnet(self):

        if not hasattr(self, 'elasticnet_info'):
            self.load_elasticnet_coefficients()
            
        # Loop over all SHAP types and plot for each
        shap_types = [
            ("5_dfs_average", f'{self.latex_symbols["Unique-Binding"]["5_dfs_average"]}', self.calculate_global_SHAP(mode="5_dfs_average", binding_unique="Unique-Binding")),
            ("Bound-Only", f"{self.latex_symbols['Unique-Binding']['Bound-Only']}", self.calculate_specialized_global_SHAP(mode="Bound-Only", condition=None, underlying_data="Unique-Binding")),
            ("NOT-Bound-Only", f"{self.latex_symbols['Unique-Binding']['NOT-Bound-Only']}", self.calculate_specialized_global_SHAP(mode="NOT-Bound-Only", condition=None, underlying_data="Unique-Binding")),
        ]

        for shap_key, shap_label, global_shap in shap_types:

            fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300, sharex=True, sharey=False)
            model_names = ["Abs(ElasticNet Coef.)", "Global SHAP"]
            cell_lines = self.cell_lines

            for row_idx, cell_line in enumerate(cell_lines):
                # Get ElasticNet coefficients and Global SHAP as 2D heatmaps
                enet_heatmap = self.convert_RBP_position_to_2d_heatmap(self.elasticnet_info[cell_line]["coefficients"]).abs()
                shap_heatmap = global_shap[cell_line]

                for col_idx, (model, heatmap) in enumerate(zip(model_names, [enet_heatmap, shap_heatmap])):

                    # Prepare data: group positions 1,2,5,6 as "1,2,5,6" and 3,4 as "3,4"
                    values = []
                    group_labels = []
                    for pos in heatmap.index:
                        for rbp in heatmap.columns:
                            val = heatmap.at[pos, rbp]
                            if pos in [1, 2, 5, 6]:
                                group_labels.append("1, 2, 5, 6")
                            elif pos in [3, 4]:
                                group_labels.append("3, 4")
                            else:
                                sys.exit(f"Unexpected position {pos} in heatmap index")

                            values.append(val)

                    plot_df = pd.DataFrame({"Value": values, "Position": group_labels})

                    if shap_key == "Bound-Only":
                        # Filter out NaN values for Bound-Only Global SHAP
                        plot_df = plot_df.dropna(subset=["Value"])

                    # Simpler violinplot and boxplot with contrasting colors and green outliers
                    ax = axes[row_idx, col_idx]
                    sns.violinplot(
                        data=plot_df,
                        x="Position",
                        y="Value",
                        ax=ax,
                        inner=None,
                        density_norm="width",
                        color="cornflowerblue",
                        linewidth=1,
                        alpha=0.4,
                        cut=0  # Prevent KDE from extending beyond the data range (no negative values)
                    )
                    sns.boxplot(
                        data=plot_df,
                        x="Position",
                        y="Value",
                        ax=ax,
                        width=0.2,
                        boxprops={"facecolor": "none", "edgecolor": "black", "zorder": 2},
                        showcaps=True,
                        showfliers=True,
                        flierprops={"marker": "o", "color": "green", "markerfacecolor": "red", "markersize": 3, "alpha": 0.3},  # smaller dots, less alpha
                        showmeans=True,
                        meanline=True,
                        meanprops={"color": "gold", "linewidth": 2}
                    )

                    # Mann-Whitney U test: test if positions 3,4 have higher values than 1,2,5,6
                    group1 = plot_df[plot_df["Position"] == "1, 2, 5, 6"]["Value"]
                    group2 = plot_df[plot_df["Position"] == "3, 4"]["Value"]
                    p_val = mannwhitneyu(group2, group1, alternative="greater", nan_policy="raise").pvalue

                    # Annotate significance bar and p-value, adjust ylim to make more room
                    y_max = plot_df["Value"].max()
                    y_min = plot_df["Value"].min()
                    y_range = y_max - y_min
                    y_bar = y_max + 0.12 * y_range   # Move bar just above the top, with extra space
                    y_text = y_bar + 0.08 * y_range  # Place annotation further above the bar

                    # Draw the significance bar
                    ax.plot([0, 0, 1, 1], [y_bar, y_bar + 0.06*y_range, y_bar + 0.06*y_range, y_bar], lw=1.5, c='k')

                    # Always center annotation at x=0.5; combine asterisk and p-value if significant
                    annotation = f"* p={p_val:.2e}" if p_val < 0.05 else f"p={p_val:.2e}"
                    ax.text(0.5, y_text, annotation, ha='center', va='bottom', fontsize=14, color="black")

                    # Extend ylim to make more room for annotation
                    current_ylim = ax.get_ylim()
                    new_ylim = (current_ylim[0], y_text + 0.15 * y_range)
                    ax.set_ylim(new_ylim)

                    # Annotate the difference of medians between "3, 4" and "1, 2, 5, 6" at (x=0.5, y=0.6) in green
                    median_difference = plot_df[plot_df['Position']=='3, 4']['Value'].median() - plot_df[plot_df['Position']=='1, 2, 5, 6']['Value'].median()
                    ax.text(
                        0.5, 0.6,
                        f"Δ Median =\n{median_difference:.2e}",
                        ha="center", va="center", fontsize=16, color="green", transform=ax.transAxes
                    )

                    ax.set_xlabel("")

                    # Set y-axis label based on model
                    if model == "Global SHAP":
                        y_label = shap_label
                    elif model == "Abs(ElasticNet Coef.)":
                        y_label = self.latex_symbols["ElasticNet Coefficients"]["Absolute Value"]
                    ax.set_ylabel(y_label, fontsize=20)

                    ax.set_title(f"{cell_line} - {y_label}", fontsize=18, pad=10)


            for ax in axes.flat:
                ax.tick_params(axis='x', labelsize=16)
                ax.tick_params(axis='y', labelsize=13)

            fig.supxlabel("Positions", fontsize=20, y=0.02)
            plt.suptitle(
                f"{shap_label}/Abs(ElasticNet Coef.): Positions 3 & 4 vs. All Other Positions\n\n"
                "NOTE 1: Absolute value used for ElasticNet coef.\n"
                f"NOTE 2: MWU test checks '3 and 4' greater than others (one-sided)\n"
                f"NOTE 3: Using {shap_label}\n"
                "NOTE 4: Δ Median is Med('3, 4') - Med('1, 2, 5, 6')",
                fontsize=16, y=1.025)
            plt.tight_layout()

            plt.savefig(self.FIGURES["position_3_4_global_shap_beta_coeff_violinplot"]["grouped_positions"][shap_key], dpi=600, bbox_inches='tight')
            plt.show()

            # New figure: violinplot and boxplot for each position (1-6) per cell line and model
            fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300, sharex=True, sharey=False)
            for row_idx, cell_line in enumerate(cell_lines):
                enet_heatmap = self.convert_RBP_position_to_2d_heatmap(self.elasticnet_info[cell_line]["coefficients"]).abs()
                shap_heatmap = global_shap[cell_line]

                for col_idx, (model, heatmap) in enumerate(zip(model_names, [enet_heatmap, shap_heatmap])):
                    # Prepare data: group by position (1-6)
                    values = []
                    positions = []
                    for pos in sorted(heatmap.index):
                        for rbp in heatmap.columns:
                            values.append(heatmap.at[pos, rbp])
                            positions.append(pos)
                    plot_df = pd.DataFrame({"Value": values, "Position": positions})
                    plot_df = plot_df.sort_values(by="Position")

                    ax = axes[row_idx, col_idx]
                    sns.violinplot(
                        data=plot_df,
                        x="Position",
                        y="Value",
                        ax=ax,
                        inner=None,
                        density_norm="width",
                        color="cornflowerblue",
                        linewidth=1,
                        alpha=0.4, 
                        cut=0  # Prevent KDE from extending beyond the data range 
                    )
                    sns.boxplot(
                        data=plot_df,
                        x="Position",
                        y="Value",
                        ax=ax,
                        width=0.2,
                        boxprops={"facecolor": "none", "edgecolor": "black", "zorder": 2},
                        showcaps=True,
                        showfliers=True,
                        flierprops={"marker": "o", "color": "green", "markerfacecolor": "red", "markersize": 3, "alpha": 0.3},
                        showmeans=True,
                        meanline=True,
                        meanprops={"color": "gold", "linewidth": 2}
                    )

                    # Set y-axis label based on model
                    if model == "Global SHAP":
                        y_label = shap_label
                    elif model == "Abs(ElasticNet Coef.)":
                        y_label = self.latex_symbols["ElasticNet Coefficients"]["Absolute Value"]
                    ax.set_ylabel(y_label, fontsize=20)

                    ax.set_title(f"{cell_line} - {y_label}", fontsize=18, pad=10)
                    ax.set_xlabel("")

                    ax.tick_params(axis='x', labelsize=20)
                    ax.tick_params(axis='y', labelsize=14)

            fig.supxlabel("Position", fontsize=22, y=0.01)
            plt.suptitle(
                f"{shap_label}/Abs(ElasticNet Coefficient) by Position\n"
                "NOTE: Absolute value used for ElasticNet coefficients.\n"
                f"NOTE 2: Using {shap_label}",
                fontsize=16, y=1.01
            )
            plt.tight_layout()

            plt.savefig(self.FIGURES["position_3_4_global_shap_beta_coeff_violinplot"]["separated_positions"][shap_key], dpi=600, bbox_inches='tight')
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
                binding_percentage = (binding_sum / num_rows)*100
                # Transpose binding_percentage and binding_sum, reset their index
                binding_percentage_long = binding_percentage.to_pandas().transpose().reset_index()
                binding_percentage_long.columns = ["Feature", "Binding Percentage"]

                binding_sum_long = binding_sum.to_pandas().transpose().reset_index()
                binding_sum_long.columns = ["Feature", "Total Binding"]

                # Remove the "_binding" suffix from the Feature column in both DataFrames
                binding_percentage_long["Feature"] = binding_percentage_long["Feature"].str.replace("_binding", "", regex=False)
                binding_sum_long["Feature"] = binding_sum_long["Feature"].str.replace("_binding", "", regex=False)

                # Merge binding_percentage_long and binding_sum_long on "Feature"
                binding_metrics_long = binding_percentage_long.merge(binding_sum_long, how="inner", on="Feature")

                summary_table_length_original = len(summary_table)
                # Ensure a 1-to-1 inner merge with the summary table
                summary_table = summary_table.merge(binding_metrics_long, how="inner", on="Feature")
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

                kd_rbps = df.filter(pl.col("RBP_KD_Target") != "CTRL").select("RBP_KD_Target").unique().collect()["RBP_KD_Target"].to_list()
                no_kd_rbps = set()
                for rbp in summary_table["RBP"].unique():
                    # IMPORTANT FIX: check if the RBP actually had knockdown data in the first place
                    if rbp not in kd_rbps: 
                        no_kd_rbps.add(rbp)
                        continue

                    num_diff_events = len(joined_df.filter(pl.col("RBP_KD_Target") == rbp))
                    num_diff_events_with_rbp_kd = len(
                        joined_df.filter((pl.col("RBP_KD_Target") == rbp) & (pl.col("has_RBP_KD") == True))
                    )
                    summary_table.loc[summary_table["RBP"] == rbp, "# Diff. Events"] = num_diff_events
                    summary_table.loc[summary_table["RBP"] == rbp, "# Diff. Events + Binding (Any Pos.)"] = num_diff_events_with_rbp_kd

                    for position in range(1,7): 
                        feature = f"{rbp}_{position}"
                        num_diff_events_with_rbp_kd_in_position = len(
                            joined_df.filter(
                                (pl.col("RBP_KD_Target") == rbp) & 
                                (pl.col(f"has_RBP_KD_{position}") == True) 
                            )
                        )        

                        summary_table.loc[summary_table["Feature"] == feature, "# Diff. Events + Binding (Specific Pos.)"] = num_diff_events_with_rbp_kd_in_position        
                
                logger.warning(f"RBPs with no knockdown data in {cell_line}: {', '.join(sorted(no_kd_rbps))}\nLength: {len(no_kd_rbps)}")
                # Add the summary table to the list
                summary_tables.append(summary_table)

            # Concatenate all summary tables for each cell line
            summary_table = pd.concat(summary_tables, ignore_index=True)

            # Create new columns for percentages using a loop
            for suffix in ["Any Pos.", "Specific Pos."]:
                summary_table[f"% Diff. Events + Binding ({suffix})"] = (
                    summary_table[f"# Diff. Events + Binding ({suffix})"] / summary_table["# Diff. Events"]
                ) * 100

            # Create "Total RBP Binding" column: sum Total Binding for each RBP within each cell line
            summary_table["Total RBP Binding"] = summary_table.groupby(["Cell Line", "RBP"])["Total Binding"].transform("sum")
            
            # Move "Total RBP Binding" column to be next to "Total Binding"
            cols = summary_table.columns.tolist()
            cols.insert(cols.index("Total Binding") + 1, cols.pop(cols.index("Total RBP Binding")))
            summary_table = summary_table[cols]

            # Create binding-normalized columns as percentages
            summary_table["Binding Norm. Ratio Diff Events"] = (summary_table["# Diff. Events"] / summary_table["Total RBP Binding"]) 
            summary_table["Binding Norm. Ratio Diff Events + Binding (Any Pos.)"] = (summary_table["# Diff. Events + Binding (Any Pos.)"] / summary_table["Total RBP Binding"]) 
            summary_table["Binding Norm. Ratio Diff Events + Binding (Specific Pos.)"] = (summary_table["# Diff. Events + Binding (Specific Pos.)"] / summary_table["Total Binding"])

            # Sort the summary table by Cell Line and Feature
            summary_table = summary_table.sort_values(by=["Cell Line", "Feature"])
            # Reset the index of the summary table
            summary_table.reset_index(drop=True, inplace=True)

            # Save the summary table to a TSV file
            summary_table.to_csv(self.CACHE_INFO["feature_metric_summary_table"], sep="\t", index=False)
            logger.success("Created feature metric summary table")
            return summary_table

    
    def get_feature_metric_table_without_null_differential_stats(self): 
        # Check if the summary table already exists
        if not hasattr(self, 'feature_metric_summary_table'):
            self.create_feature_metric_summary_table()
        
        table = copy.deepcopy(self.feature_metric_summary_table)
        table = table[table["# Diff. Events"].notnull()]
        
        # Allow nulls only in columns that begin with "Binding Normalized %"
        allowed_null_prefix = "Binding Norm. Ratio"
        cols_with_nulls = table.columns[table.isnull().any()].tolist()
        disallowed_nulls = [col for col in cols_with_nulls if not col.startswith(allowed_null_prefix)]
        if disallowed_nulls:
            raise AssertionError(
            f"Null values found in columns other than those starting with '{allowed_null_prefix}': {disallowed_nulls}"
            )

        return table
    

    def plot_feature_metric_summary_table(self):
        if not hasattr(self, 'feature_metric_summary_table'):
            self.create_feature_metric_summary_table()
        
        for plotting_column in self.binding_normalized_differential_plotting_columns_info:
            logger.info(f"Plotting {plotting_column} for each cell line")
            for mode in ["All Features", "Only RBPs"]:

                if plotting_column == "# Diff. Events + Binding (Specific Pos.)" and mode == "Only RBPs":
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

        for plotting_column, specificity in self.non_normalized_differential_plotting_columns_info.items():
            logger.info(f"Processing {plotting_column} ({specificity})")
            metric_summary_table = self.get_feature_metric_table_without_null_differential_stats()

            if specificity == "RBP-specific":
                metric_summary_table = metric_summary_table.drop_duplicates(subset=["Cell Line", "RBP"], keep="first")
            elif specificity == "Feature-specific":
                pass  # Use the entire table

            self.plot_diff_events_histogram_boxplot(metric_summary_table, plotting_column, specificity)
        
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

            self.plot_diff_events_heatmaps(metric_summary_table, plotting_column, specificity, matching_data)

            self.plot_diff_events_matching_scatterplot(matching_data, plotting_column, specificity)            

    
    def plot_diff_events_histogram_boxplot(self, metric_summary_table, plotting_column, specificity): 

        logger.info(f"Processing {plotting_column} ({specificity})")

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
                    -0.05,  # Use the index or a unique identifier for labeling
                    row["RBP"] if specificity == "RBP-specific" else row["Feature"],
                    fontsize=4,
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


    def plot_diff_events_heatmaps(self, metric_summary_table, plotting_column, specificity, matching_data):
        
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

                # Always annotate with integer values from the table
                annot = heatmap_data.astype(int)
                fmt = "d"

                # Plot heatmap for the cell line
                sns.heatmap(
                    heatmap_data,
                    ax=ax,
                    cmap="viridis",
                    cbar=True,
                    linewidths=0.5,
                    linecolor="gray",
                    cbar_kws={"shrink": 0.8, "aspect": 5, "pad": 0.01},  # Adjust colorbar position and size
                    norm=norm,  # Apply log scale to the colorbar if specified
                    annot=annot,
                    fmt=fmt,  # Use the annotation as is
                    annot_kws={"size": 17, "rotation": 90},
                )
                cbar = ax.collections[0].colorbar  # Get the colorbar
                cbar.ax.tick_params(labelsize=12)  # Set the font size of the colorbar ticks

                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_title(f"{cell_line} (# RBPs = {len(heatmap_data.columns)})", fontsize=24)

                if specificity == "Feature-specific":
                    ax.tick_params(axis='y', labelsize=24)
                elif specificity == "RBP-specific":
                    ax.tick_params(axis='y', labelleft=False)

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

        # Determine global min and max values for consistent color scaling
        vmin = min(matching_data[cell_line].min().min() for cell_line in self.cell_lines)
        vmax = max(matching_data[cell_line].max().max() for cell_line in self.cell_lines)

        for log_transform in [False, True]:
            fig, axes = plt.subplots(2, 1, figsize=(30, y_fig_size), dpi=300, sharex=True)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position for the single colorbar

            # Define a single norm for both cell lines
            norm = LogNorm(vmin=vmin+1, vmax=vmax) if log_transform else None

            for ax, cell_line in zip(axes, self.cell_lines):
                # Always annotate with integer values from the table
                annot = matching_data[cell_line].astype(int)
                fmt = "d"

                sns.heatmap(
                    matching_data[cell_line],
                    cmap="viridis",
                    cbar=(ax == axes[0]),  # Add colorbar only for the first heatmap
                    cbar_ax=(cbar_ax if ax == axes[0] else None),
                    linewidths=0.5,
                    linecolor="gray",
                    norm=norm,  # Use the shared norm for both heatmaps
                    vmin=vmin if not log_transform else None,  # Explicitly pass vmin
                    vmax=vmax if not log_transform else None,  # Explicitly pass vmax
                    ax=ax,
                    annot=annot,
                    fmt=fmt,  # Use the annotation as is
                    annot_kws={"size": 25, "rotation": 90},
                )
                ax.set_title(f"{cell_line} {'Ordered' if cell_line == self.cell_lines[0] else 'Matching'} (# RBPs = {len(matching_data[cell_line].columns)})", fontsize=24)
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


    def plot_diff_events_matching_scatterplot(self, matching_data, plotting_column, specificity):
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
                row[f"{self.cell_lines[0]} {plotting_column}"] - 10,
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
            0.6, 0.97,
            f"Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}\nPoints: {len(merged_data)}",
            transform=plt.gca().transAxes,
            fontsize=8,
            verticalalignment="top"
        )
        plt.tight_layout()
        plt.show()


    def plot_differential_ranks(self): 

        for column_suffix in ["Any Pos.", "Specific Pos."]:
            copy_df = self.get_feature_metric_table_without_null_differential_stats()
            rank_column = f"# Diff. Events + Binding ({column_suffix})"

            if column_suffix == "Any Pos.":
                # Take unique values by both Cell Line and RBP
                summary_table = copy_df.drop_duplicates(subset=["Cell Line", "RBP"])
            else: 
                summary_table = copy_df
                

            # Create a figure with 2 rows (one for each cell line) and 1 column
            fig, axes = plt.subplots(2, 1, figsize=(5,7), dpi=300, sharex=False, sharey=False)

            for row_idx, cell_line in enumerate(self.cell_lines):
                # Subset the data for the specific cell line
                cell_line_data = summary_table[summary_table["Cell Line"] == cell_line].copy()

                # Rank the columns of interest
                cell_line_data["Rank: # Diff. Events"] = cell_line_data["# Diff. Events"].rank(ascending=False)
                cell_line_data[f"Rank: {rank_column}"] = cell_line_data[rank_column].rank(ascending=False)

                # Calculate the change in ranks
                cell_line_data["Rank Change"] = abs(
                    cell_line_data["Rank: # Diff. Events"] - cell_line_data[f"Rank: {rank_column}"]
                )

                # Scatterplot of the ranks
                ax = axes[row_idx]
                sns.scatterplot(
                    data=cell_line_data,
                    x="Rank: # Diff. Events",
                    y=f"Rank: {rank_column}",
                    alpha=0.7,
                    edgecolor="black",
                    color="deepskyblue",
                    s=20,
                    ax=ax
                )

                # Add a diagonal line for reference
                ax.plot(
                    [cell_line_data["Rank: # Diff. Events"].min(), cell_line_data["Rank: # Diff. Events"].max()],
                    [cell_line_data["Rank: # Diff. Events"].min(), cell_line_data["Rank: # Diff. Events"].max()],
                    color="red",
                    linestyle="--",
                    linewidth=1,
                    label="y=x"
                )

                # Annotate the top 5 rank changes
                top_5 = cell_line_data.nlargest(20, "Rank Change")
                for _, row in top_5.iterrows():
                    ax.text(
                    row["Rank: # Diff. Events"] - 2,
                    row[f"Rank: {rank_column}"] + 2,
                    row["RBP"],
                    fontsize=3,
                    color="black",
                    alpha=0.8
                    )

                spearman_corr, _ = spearmanr(
                    cell_line_data["Rank: # Diff. Events"],
                    cell_line_data[f"Rank: {rank_column}"]
                )

                # Add labels, title, and legend
                ax.set_title(f"{cell_line}", fontsize=12)
                ax.set_xlabel("")
                ax.set_ylabel("")

                # Add correlation and point count in the bottom right corner
                num_points = len(cell_line_data)
                ax.text(
                    0.99, 0.03,
                    f"Spearman: {spearman_corr:.2f}\nPoints: {num_points}",
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment="bottom",
                    horizontalalignment="right"
                )

            # Adjust layout and show the plot
            plt.tight_layout()

            if column_suffix == "Any Pos.":
                plt.suptitle(f"Rank Comparison: # Diff Events Total vs Binding ({column_suffix})\nNOTE: this is RBP-specific\nNOTE 2: Rank 1 is highest", fontsize=10, y=1.06)
            elif column_suffix == "Specific Pos.":
                plt.suptitle(f"Rank Comparison: # Diff Events Total vs Binding ({column_suffix})\nNOTE: this is Feature-specific but x-axis is RBP-specific\nNOTE 2: Rank 1 is highest", fontsize=10, y=1.07)

            fig.supxlabel("Rank: # Diff. Events", fontsize=10, y=-0.01)
            fig.supylabel(f"Rank: {rank_column}", fontsize=10, x=-0.01)
            plt.show()


    def plot_differential_ratios(self): 

        for plotting_column, specificity in self.normalized_differential_plotting_columns_info.items():
            logger.info(f"Processing {plotting_column} ({specificity})")
            data = self.get_feature_metric_table_without_null_differential_stats()

            # Create plotting_data dictionary
            plotting_data = {}
            if specificity == "RBP-specific":
                
                data = data.drop_duplicates(subset=["Cell Line", "RBP"])
                for cell_line in self.cell_lines:
                    cell_line_data = data[data["Cell Line"] == cell_line]
                    plotting_data[cell_line] = cell_line_data.sort_values(by=plotting_column, ascending=True).set_index("RBP")[[plotting_column]].T

            elif specificity == "Feature-specific":
                for cell_line in self.cell_lines:
                    cell_line_data = data[data["Cell Line"] == cell_line].pivot(index="Position", columns="RBP", values=plotting_column)
                    linkage = sch.linkage(cell_line_data.T, method="ward")
                    dendrogram = sch.dendrogram(linkage, no_plot=True)
                    ordered_columns = [cell_line_data.columns[i] for i in dendrogram["leaves"]]
                    plotting_data[cell_line] = cell_line_data[ordered_columns]

            # Determine global min and max values for consistent color scaling
            vmin = min(plotting_data[cell_line].min().min() for cell_line in self.cell_lines)
            vmax = max(plotting_data[cell_line].max().max() for cell_line in self.cell_lines)
            
            if specificity == "RBP-specific":
                y_fig_size = 9
            elif specificity == "Feature-specific":
                y_fig_size = 19

            # Plot heatmaps for both cell lines
            fig, axes = plt.subplots(2, 1, figsize=(30, y_fig_size), dpi=300, sharex=False)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position for the single colorbar

            for ax, cell_line in zip(axes, self.cell_lines):
                sns.heatmap(
                    plotting_data[cell_line],
                    cmap="Blues",  # Use the Blues colormap
                    cbar=(ax == axes[0]),  # Add colorbar only for the first heatmap
                    cbar_ax=(cbar_ax if ax == axes[0] else None),
                    linewidths=0.5,
                    linecolor="black",  # Set cell borders to black
                    vmin=vmin,
                    vmax=vmax,
                    annot=plotting_data[cell_line].round(1),  # Annotate with values rounded to 1 decimal place
                    fmt=".1f",  # Format annotations to 1 decimal place
                    annot_kws={"size": 15, "rotation": 90},  # Rotate and enlarge annotation text
                    ax=ax
                )
                ax.set_title(f"{cell_line} (# RBPs = {len(plotting_data[cell_line].columns)})", fontsize=20)
                ax.set_xlabel("")
                ax.set_ylabel("")

                if specificity == "RBP-specific":
                    ax.tick_params(axis='y', labelleft=False)  # Remove y tick labels
                elif specificity == "Feature-specific":
                    ax.tick_params(axis='y', labelsize=24)

            # Add colorbar title
            cbar_ax.set_title("%", fontsize=15)
            cbar_ax.tick_params(labelsize=12)

            fig.supxlabel("RBP", fontsize=20, x=0.45)
            fig.supylabel("Cell Line" if specificity == "RBP-specific" else "Position", fontsize=20, x=-0.01)

            plt.suptitle(
                f"Heatmap of {plotting_column}\nNOTE: Different RBPs on each x-axis\nNOTE 2: {specificity} \n"
                f"NOTE 3: {'Sorted by RBP Value' if specificity == 'RBP-specific' else 'Ward Hierarchical Clustering'}",
                fontsize=25, y=1.02
            )
            plt.tight_layout(rect=[0, 0, 0.91, 1])  # Adjust layout to make space for the colorbar
            plt.show()


    def plot_matching_differential_ratios_scatterplot(self):

        if not hasattr(self, 'feature_metric_summary_table'):
            self.create_feature_metric_summary_table()
        
        # Get matching features across both cell lines
        matching_features = self.get_matching_features()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

        for i, (plotting_column, specificity) in enumerate(self.normalized_differential_plotting_columns_info.items()):
            # Subset the summary table to matching features only
            df = self.feature_metric_summary_table[
                self.feature_metric_summary_table["Feature"].isin(matching_features)
            ]

            # Pivot to get values for both cell lines side by side
            pivot = df.pivot(index="Feature", columns="Cell Line", values=plotting_column).dropna()
            assert pivot.notnull().all().all(), "Null or missing values found in the pivot table"

            # If RBP-specific, deduplicate by RBP extracted from Feature
            if specificity == "RBP-specific":
                # Extract RBP from Feature (e.g., "RBP_1" -> "RBP")
                pivot = pivot.reset_index()
                pivot["RBP"] = pivot["Feature"].str.split("_").str[0]
                # Subset to unique RBPs (keep first occurrence)
                pivot = pivot.drop_duplicates(subset="RBP").set_index("RBP")

            x = pivot[self.cell_lines[0]]
            y = pivot[self.cell_lines[1]]

            # Calculate correlations
            pearson_corr, _ = pearsonr(x, y)
            spearman_corr, _ = spearmanr(x, y)
            num_points = len(pivot)

            # Scatterplot
            ax = axes[i]
            sns.scatterplot(x=x, y=y, ax=ax, color="deepskyblue", edgecolor="black", alpha=0.7, s=30)
            ax.plot([x.min(), x.max()], [x.min(), x.max()], color="red", linestyle="--", linewidth=1, label="y=x")
            ax.set_xlabel(f"{self.cell_lines[0]} {plotting_column}", fontsize=10)
            ax.set_ylabel(f"{self.cell_lines[1]} {plotting_column}", fontsize=10)
            ax.set_title(f"Matching Features: {plotting_column}", fontsize=12)
            ax.text(
            0.98, 0.5,
            f"Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}\nPoints: {num_points}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            horizontalalignment='right'
            )

            # Label the top 10 points on either axis using their index as the label
            top_10_x = x.nlargest(10)
            top_10_y = y.nlargest(10)
            top_indices = set(top_10_x.index).union(set(top_10_y.index))
            for idx in top_indices:
                ax.text(
                    x[idx],
                    y[idx],
                    str(idx),
                    fontsize=6,
                    color="black",
                    alpha=0.8
                )


        plt.suptitle("% Diff. Splicing Metrics for Matching Features\nNOTE: when plotting percentages")
        plt.tight_layout()
        plt.show()


    def get_matching_features(self):

        if not hasattr(self, 'feature_metric_summary_table'):
            self.create_feature_metric_summary_table()
        
        # Copy the feature metric summary table
        summary_table = self.feature_metric_summary_table.copy()

        # Get unique features for each cell line
        features_per_cell_line = {
            cell_line: set(summary_table[summary_table["Cell Line"] == cell_line]["Feature"].unique())
            for cell_line in self.cell_lines
        }
        assert len(features_per_cell_line) == len(self.cell_lines), "Mismatch in number of cell lines"

        return set.intersection(*features_per_cell_line.values())
    

    def get_local_SHAP_based_on_binding_and_covariates(self, binding_value, binding_pattern_type=None):

        assert binding_pattern_type in ["All-Data", "Unique-Binding"], "binding_pattern_type must be 'All-Data' or 'Unique-Binding'"
        assert binding_value in [0, 1], "binding_value must be 0 or 1"
        assert not hasattr(self, 'final_all_data_SHAP_data') and not hasattr(self, 'final_unique_binding_SHAP_data'), "Ensure final_all_data_SHAP_data and final_unique_binding_SHAP_data are not already set"

        final_shap_lazyframes = self.load_final_SHAP_data(underlying_data=binding_pattern_type, as_lazyframe=True)

        for cell_line in self.cell_lines:
            shap_lazyframe = final_shap_lazyframes[cell_line]
            binding_cols = [col for col in shap_lazyframe.collect_schema().names() if col.endswith("_binding")]
            shap_cols = [col for col in shap_lazyframe.collect_schema().names() if col.endswith("_shap")]

            df = shap_lazyframe.select(binding_cols + shap_cols).collect()

            for binding_col in tqdm.tqdm(binding_cols, desc=f"{cell_line} binding_cols"):
                shap_col = binding_col.replace("_binding", "_shap")
                series = (
                    df
                    .select([binding_col, shap_col])
                    .filter(pl.col(binding_col) == binding_value)
                    .select(shap_col)[shap_col]
                )
                yield (cell_line, shap_col, series)

    
    def calculate_specialized_global_SHAP(self, mode=None, condition=None, underlying_data=None): 
        VALID_MODES = ["Bound-Only", "NOT-Bound-Only", "Signed-Local-SHAP-Mean-Bound-Only", "Signed-Local-SHAP-Mean-NOT-Bound-Only"]
        assert mode in VALID_MODES, f"Invalid mode. Choose from {VALID_MODES}"
        assert condition in [None, "CTRL", "RBP_KD", 'RBP_KD_at_position'], "Condition must be None, 'CTRL', 'RBP_KD', or 'RBP_KD_at_position'"
        assert underlying_data in ["All-Data", "Unique-Binding"], "underlying_data must be 'All-Data' or 'Unique-Binding'"

        OUTPUT_FILE = self.CACHE_INFO["specialized_global_SHAP"][mode][condition][underlying_data]

        if os.path.exists(OUTPUT_FILE):
            logger.success(f"FROM CACHE: loading specialized global SHAP for mode {mode}, condition {condition}, and underlying_data {underlying_data} from {OUTPUT_FILE}")
            with open(OUTPUT_FILE, "rb") as f:
                specialized_global_SHAP = pickle.load(f)
            return specialized_global_SHAP
        
        else: 

            logger.info(f"Calculating specialized global SHAP for mode {mode}, condition {condition}, and underlying_data {underlying_data}.")
            specialized_global_SHAP = {}

            if mode in ["Bound-Only", "Signed-Local-SHAP-Mean-Bound-Only"]:
                binding_value = 1
            elif mode in ["NOT-Bound-Only", "Signed-Local-SHAP-Mean-NOT-Bound-Only"]:
                binding_value = 0

            # Use the generator to collect mean SHAP values for each feature at the given binding value
            cell_line_feature_dict = {cell_line: {} for cell_line in self.cell_lines}
            for yielded_cell_line, shap_col, series in self.get_local_SHAP_based_on_binding_and_covariates(binding_value, binding_pattern_type=underlying_data):
                
                if mode.startswith("Signed-Local-SHAP-Mean"):
                    tmp_mean = series.mean()
                else: 
                    tmp_mean = series.abs().mean()

                cell_line_feature_dict[yielded_cell_line][shap_col] = tmp_mean

            for cell_line in self.cell_lines:
                results = cell_line_feature_dict[cell_line]
                specialized_df = pd.DataFrame([results])
                specialized_df = self.convert_RBP_position_to_2d_heatmap(specialized_df)
                specialized_global_SHAP[cell_line] = specialized_df

            # Save the specialized global SHAP to cache
            with open(OUTPUT_FILE, "wb") as f:
                pickle.dump(specialized_global_SHAP, f)
            
            logger.success(f"Specialized global SHAP for mode {mode}, condition {condition}, and underlying_data {underlying_data} saved to {OUTPUT_FILE}")
            return specialized_global_SHAP


    def plot_specialized_vs_regular_global_SHAP(self, underlying_data=None):
        assert underlying_data == "Unique-Binding", "Currently only 'Unique-Binding' is supported for underlying_data"
        
        # Load all global SHAP variants
        global_shap_regular = self.calculate_global_SHAP(mode="5_dfs_average", binding_unique = underlying_data)
        global_shap_bound = self.calculate_specialized_global_SHAP(mode="Bound-Only", condition=None, underlying_data= underlying_data)
        global_shap_not_bound = self.calculate_specialized_global_SHAP(mode="NOT-Bound-Only", condition=None, underlying_data= underlying_data)

        # Prepare all metric variants
        shap_variants = {
            "Everything": global_shap_regular,
            "NOT Bound Only": global_shap_not_bound,
            "Bound Only": global_shap_bound,
        }

        # Get all pairwise combinations (excluding self-comparisons)
        variant_pairs = list(combinations(shap_variants.items(), 2))

        for (label_x, data_x), (label_y, data_y) in variant_pairs:
            fig, axes = plt.subplots(1, len(self.cell_lines), figsize=(10, 5), dpi=300)

            for idx, cell_line in enumerate(self.cell_lines):

                df_x = data_x[cell_line]
                df_y = data_y[cell_line]

                # Melt both DataFrames to long format with RBP and Position columns
                df_x_long = df_x.reset_index().melt(id_vars=df_x.index.name or "index", var_name="RBP", value_name="x_val")
                df_x_long = df_x_long.rename(columns={df_x.index.name or "index": "Position"})
                df_y_long = df_y.reset_index().melt(id_vars=df_y.index.name or "index", var_name="RBP", value_name="y_val")
                df_y_long = df_y_long.rename(columns={df_y.index.name or "index": "Position"})

                # Merge on RBP and Position to align values
                merged = pd.merge(df_x_long, df_y_long, on=["RBP", "Position"])
                # Remove NaNs
                merged = merged.dropna(subset=["x_val", "y_val"])

                # Correlations
                pearson_corr, _ = pearsonr(merged["x_val"], merged["y_val"])
                spearman_corr, _ = spearmanr(merged["x_val"], merged["y_val"])
                num_points = len(merged)

                ax = axes[idx]
                sns.scatterplot(x=merged["x_val"], y=merged["y_val"], color="deepskyblue", edgecolor="black", alpha=0.5, s=20, ax=ax)
                ax.set_title(f"{cell_line}", fontsize=14)
                ax.set_xlabel('')
                ax.set_ylabel('')

                # Annotate top 10 points on x and y axes with RBP_Position
                top_10_x = merged.nlargest(20, "x_val")
                top_10_y = merged.nlargest(20, "y_val")
                top = pd.concat([top_10_x, top_10_y]).drop_duplicates()
                for _, row in top.iterrows():
                    label = f"{row['RBP']}_{row['Position']}"
                    ax.text(
                        row["x_val"],
                        row["y_val"],
                        label,
                        fontsize=4,
                        color="black",
                        alpha=1
                    )

                ax.text(
                    0.98, 0.02,
                    f"Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}\nPoints: {num_points}",
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='bottom',
                    horizontalalignment='right'
                )

            plt.suptitle(f"Global SHAP: {label_x} vs {label_y}\nNOTE: using {underlying_data} data", fontsize=16, y=1.02)
            fig.supxlabel(f"Global SHAP: {label_x}", fontsize=12)
            fig.supylabel(f"Global SHAP: {label_y}", fontsize=12)
            plt.tight_layout()
            plt.show()


    def shap_for_zero_binding_features_by_condition(self): 

        # Get specialized global SHAP for NOT-Bound-Only and all valid conditions
        mode = "NOT-Bound-Only"
        valid_conditions = ["CTRL", "RBP_KD", "RBP_KD_at_position"]
        # Get all specialized SHAP heatmaps for each condition
        specialized_shap = {cond: self.calculate_specialized_global_SHAP(mode=mode, condition=cond) for cond in valid_conditions}

        for cell_line in self.cell_lines:
            # Gather all heatmaps for this cell line to compute global min/max
            heatmaps = [specialized_shap[cond][cell_line] for cond in valid_conditions]
            all_values = np.concatenate([h.values.flatten() for h in heatmaps])
            all_values = all_values[~np.isnan(all_values)]
            vmin, vmax = all_values.min(), all_values.max()

            for log_scale in [False, True]:
                fig, axes = plt.subplots(3, 1, figsize=(35, 28), dpi=300, sharex=True)
                cbar_ax = fig.add_axes([0.94, 0.1, 0.02, 0.6])

                # Use hierarchical clustering on "CTRL" condition to get column order
                base_heatmap = specialized_shap["CTRL"][cell_line]
                linkage = sch.linkage(base_heatmap.T, method="ward")
                dendro = sch.dendrogram(linkage, no_plot=True)
                ordered_cols = [base_heatmap.columns[i] for i in dendro["leaves"]]

                for idx, cond in enumerate(valid_conditions):
                    data = specialized_shap[cond][cell_line][ordered_cols]
                    norm = LogNorm(vmin=max(vmin, 1e-8), vmax=vmax) if log_scale else None

                    sns.heatmap(
                        data,
                        ax=axes[idx],
                        cmap="Blues",
                        cbar=(idx == 0),
                        cbar_ax=(cbar_ax if idx == 0 else None),
                        vmin=vmin,
                        vmax=vmax,
                        norm=norm,
                        linewidths=0.5,
                        linecolor="gray",
                        annot=True,
                        fmt=".3f",
                        annot_kws={"size": 16, "rotation": 90},
                    )

                    # Always set the face color to black
                    axes[idx].set_facecolor("black")
                    
                    if cond == "CTRL":
                        subplot_title = "CTRL (RBPs clustered by Ward)"
                    else:
                        subplot_title = " (Matching RBP clustering order from CTRL)"
                        if cond == "RBP_KD_at_position":
                            subplot_title = "RBP KD at SPECIFIC Position" +  subplot_title
                        elif cond == "RBP_KD":
                            subplot_title = "RBP KD (regardless of KD RBP being originally bound to any position)" +  subplot_title

                    axes[idx].set_title(f"{subplot_title}", fontsize=30)
                    axes[idx].set_xlabel("")
                    axes[idx].set_ylabel("")
                    axes[idx].tick_params(axis='y', labelsize=24)
                    axes[idx].tick_params(axis='x', labelsize=16)

                cbar_ax.set_title('"Global SHAP"', fontsize=30, pad=20)
                cbar_ax.tick_params(labelsize=24)
                
                fig.supxlabel("RBP", fontsize=40, x=0.45)
                fig.supylabel("Position", fontsize=40, x=-0.01, y=0.35)
                plt.suptitle(
                    f"{cell_line}: 'NOT Bound Global SHAP' for [1] CTRL, [2] RBP KD, and [3] RBP KD (Specific Position)\n\nNOTE 1: 'RBP KD' takes all events from that feature's RBP KD sample (DOESN'T matter if KD RBP was bound to any position)\n\nNOTE 2: 'CTRL' RBP order clustered by Ward and all other heatmaps match RBP order\n\nNOTE 3: Black square indicates either [1] 'Feature\'s RBP never had KD experiment' or\n[2] 'Feature\'s RBP had KD experiment but no examples found' or\n[3] (IF APPLICABLE) Log of Zero\n\nNOTE 4: colorbar shared across all heatmaps\n\n{'NOTE 5: colors in log-scale (only shows values > 0)' if log_scale else ''}\n",
                    fontsize=35, y=0.99
                )
                plt.tight_layout(rect=[0, 0, 0.91, 1])
                plt.show()

        # Take every 2-length combination of valid_conditions
        condition_pairs = list(combinations(valid_conditions, 2))

        for cell_line in self.cell_lines:
            fig, axes = plt.subplots(len(condition_pairs), 1, figsize=(4, 8), dpi=300)
            for ax, (cond1, cond2) in zip(axes, condition_pairs):
                # Get the two heatmaps
                df1 = specialized_shap[cond1][cell_line]
                df2 = specialized_shap[cond2][cell_line]

                # Convert to long form
                df1_long = df1.reset_index().melt(id_vars=df1.index.name or "index", var_name="RBP", value_name=f"{cond1}_val")
                df1_long = df1_long.rename(columns={df1.index.name or "index": "Position"})
                df2_long = df2.reset_index().melt(id_vars=df2.index.name or "index", var_name="RBP", value_name=f"{cond2}_val")
                df2_long = df2_long.rename(columns={df2.index.name or "index": "Position"})

                # Merge on RBP and Position
                merged = pd.merge(df1_long, df2_long, on=["RBP", "Position"])
                # Remove rows with null values
                merged = merged.dropna(subset=[f"{cond1}_val", f"{cond2}_val"])

                x = merged[f"{cond1}_val"]
                y = merged[f"{cond2}_val"]

                # Calculate correlations
                pearson_corr, _ = pearsonr(x, y)
                spearman_corr, _ = spearmanr(x, y)
                num_points = len(merged)

                # Scatterplot
                sns.scatterplot(x=x, y=y, ax=ax, color="deepskyblue", edgecolor="black", alpha=0.7, s=10)

                ax.plot([x.min(), x.max()], [x.min(), x.max()], color="red", linestyle="--", linewidth=1, label="y=x")

                ax.set_xlabel(f"{cond1.replace('_', ' ')}", fontsize=10)
                ax.set_ylabel(f"{cond2.replace('_', ' ')}", fontsize=10)
                ax.set_title(f"{cond1.replace('_', ' ')} vs {cond2.replace('_', ' ')}", fontsize=10)

                # Annotate top 10 points on each axis
                top_10_x = merged.nlargest(10, f"{cond1}_val")
                top_10_y = merged.nlargest(10, f"{cond2}_val")
                top = pd.concat([top_10_x, top_10_y]).drop_duplicates()

                for _, row in top.iterrows():
                    label = f"{row['RBP']}_{row['Position']}"
                    ax.text(
                        row[f"{cond1}_val"],
                        row[f"{cond2}_val"],
                        label,
                        fontsize=4,
                        color="black",
                        alpha=1
                    )

                # Add stats
                ax.text(
                    0.98, 0.02,
                    f"Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}\nPoints: {num_points}",
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment='bottom',
                    horizontalalignment='right'
                )

            plt.suptitle(f"{cell_line}: 'NOT Bound Global SHAP' Pairwise Combinations for\n[1] CTRL, [2] RBP KD, and [3] RBP KD (Specific Position)", fontsize=8, y=1.0)
            plt.tight_layout()
            plt.show()


    def calculate_percent_positive_and_negative_local_SHAP_per_feature(self, mode=None, underlying_data=None):
        assert underlying_data== "Unique-Binding", "Underlying data must be 'Unique-Binding' for this function"
        assert mode in ["NOT-Bound-Only", "Bound-Only"], "Mode must be 'NOT-Bound-Only' or 'Bound-Only' for this function"
        OUTPUT_FILE = self.CACHE_INFO["local_SHAP_percent_positive_negative"][mode]

        if os.path.exists(OUTPUT_FILE):
            logger.success(f"FROM CACHE: loading percent positive and negative local SHAP for mode {mode} from {OUTPUT_FILE}")
            with open(OUTPUT_FILE, "rb") as f:
                percent_positive_and_negative_local_SHAP = pickle.load(f)
            return percent_positive_and_negative_local_SHAP
        
        else:
            logger.info(f"Calculating percent positive and negative local SHAP for mode {mode}")
            POSITIVE_NEGATIVE_CUTOFFS = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1]

            # Get local SHAP values for the entire dataset with the specified binding_value based on mode
            if mode == "NOT-Bound-Only":
                binding_value = 0
            elif mode == "Bound-Only":
                binding_value = 1
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            
            # Prepare results as: {cell_line: {shap_col: {cutoff: {"positive": val, "negative": val}}}}
            results = {cell_line: {} for cell_line in self.cell_lines}

            # Iterate through the generator once, and for each feature, compute percent positive/negative for all cutoffs
            for yielded_cell_line, shap_col, series in self.get_local_SHAP_based_on_binding_and_covariates(
                binding_value, 
                binding_pattern_type=underlying_data
            ):
                total = len(series)
                if total == 0:
                    percent_pos = {cutoff: float('nan') for cutoff in POSITIVE_NEGATIVE_CUTOFFS}
                    percent_neg = {cutoff: float('nan') for cutoff in POSITIVE_NEGATIVE_CUTOFFS}
                else:
                    arr = series.to_numpy()
                    percent_pos = {cutoff: ((np.sum(arr > cutoff)) / total) * 100 for cutoff in POSITIVE_NEGATIVE_CUTOFFS}
                    percent_neg = {cutoff: ((np.sum(arr < -cutoff)) / total) * 100 for cutoff in POSITIVE_NEGATIVE_CUTOFFS}

                # Store by cell_line and shap_col
                results[yielded_cell_line][shap_col] = {
                    cutoff: {"positive": percent_pos[cutoff], "negative": percent_neg[cutoff]}
                    for cutoff in POSITIVE_NEGATIVE_CUTOFFS
                }

            # Now, for each cutoff, build the output structure as before
            results_by_cutoff = {}
            for cutoff in POSITIVE_NEGATIVE_CUTOFFS:
                percent_positive = {cell_line: {} for cell_line in self.cell_lines}
                percent_negative = {cell_line: {} for cell_line in self.cell_lines}

                for cell_line in self.cell_lines:
                    pos_results = {}
                    neg_results = {}
                    for shap_col, cutoff_dict in results[cell_line].items():
                        rbp, pos = self.get_RBP_position(shap_col)

                        percent_pos = cutoff_dict[cutoff]["positive"]
                        percent_neg = cutoff_dict[cutoff]["negative"]

                        pos_results.setdefault(pos, {})[rbp] = percent_pos
                        neg_results.setdefault(pos, {})[rbp] = percent_neg

                    df_pos = pd.DataFrame.from_dict(pos_results, orient="index")
                    df_neg = pd.DataFrame.from_dict(neg_results, orient="index")
                    percent_positive[cell_line] = df_pos.sort_index().sort_index(axis=1)
                    percent_negative[cell_line] = df_neg.sort_index().sort_index(axis=1)

                results_by_cutoff[cutoff] = {
                    "positive": percent_positive,
                    "negative": percent_negative
                }

            with open(OUTPUT_FILE, "wb") as f:
                pickle.dump(results_by_cutoff, f)

            return results_by_cutoff


    def plot_percent_positive_and_negative_local_SHAP_per_feature(self, underlying_data=None): 
        assert underlying_data == "Unique-Binding", "Underlying data must be 'Unique-Binding' for this function"

        # Load the percent positive/negative local SHAP pickle files for Bound-Only and NOT-Bound-Only
        bound_data = self.calculate_percent_positive_and_negative_local_SHAP_per_feature(mode="Bound-Only", underlying_data=underlying_data)
        not_bound_data = self.calculate_percent_positive_and_negative_local_SHAP_per_feature(mode="NOT-Bound-Only", underlying_data=underlying_data)

        # CUTOFFS = bound_data.keys()
        CUTOFFS = [.01, .05, .1]

        for cell_line in self.cell_lines:
            for cutoff in CUTOFFS:
                bound_cut = bound_data[cutoff]
                not_bound_cut = not_bound_data[cutoff]

                # Check all NOT-Bound-Only dataframes for nulls and all dataframes for values in [0, 100] (ignoring NaNs)
                for mode_data in [bound_cut, not_bound_cut]:
                    for sign in ["positive", "negative"]:
                        for cl, df in mode_data[sign].items():
                            # Assert all NOT-Bound-Only dataframes have no nulls
                            if mode_data is not_bound_cut:
                                assert not df.isnull().values.any(), f"Nulls found in NOT-Bound-Only {sign} for {cl}"
                            # Assert all values are between 0 and 100 (ignoring NaNs)
                            arr = df.values
                            arr_no_nan = arr[~pd.isnull(arr)]
                            assert ((arr_no_nan >= 0) & (arr_no_nan <= 100)).all(), f"Values out of range in {sign} for {cl}"

        # First Figure: Heatmaps for each cutoff
        for cutoff in CUTOFFS:
            for cell_line in self.cell_lines:
                bound_cut = bound_data[cutoff]
                not_bound_cut = not_bound_data[cutoff]

                # Prepare dataframes
                df_bound_pos = bound_cut["positive"][cell_line]
                df_bound_neg = bound_cut["negative"][cell_line]
                df_not_bound_pos = not_bound_cut["positive"][cell_line]
                df_not_bound_neg = not_bound_cut["negative"][cell_line]

                # Cluster the Bound Only Positive heatmap (row 0, col 0)
                linkage = sch.linkage(df_bound_pos.fillna(0).T, method="ward")
                dendro = sch.dendrogram(linkage, no_plot=True)
                ordered_cols = [df_bound_pos.columns[i] for i in dendro["leaves"]]

                # Reorder all heatmaps to match this clustering order
                df_bound_pos = df_bound_pos[ordered_cols]
                df_bound_neg = df_bound_neg[ordered_cols]
                df_not_bound_pos = df_not_bound_pos[ordered_cols]
                df_not_bound_neg = df_not_bound_neg[ordered_cols]

                # Set up figure and axes
                fig, axes = plt.subplots(2, 2, figsize=(50, 22), dpi=300, sharex=True, sharey=True)
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Single colorbar

                heatmaps = [
                    (df_bound_pos, "Bound Only - % Positive (Clustered with Ward)", 0, 0),
                    (df_not_bound_pos, "NOT Bound Only - % Positive (cell order matches top-left heatmap)", 0, 1),
                    (df_bound_neg, "Bound Only - % Negative (cell order matches top-left heatmap)", 1, 0),
                    (df_not_bound_neg, "NOT Bound Only - % Negative (cell order matches top-left heatmap)", 1, 1),
                ]

                vmin, vmax = 0, 100

                for idx, (data, title, row, col) in enumerate(heatmaps):
                    ax = axes[row, col]
                    sns.heatmap(
                        data,
                        ax=ax,
                        cmap="Reds" if "% Positive" in title else "Blues",
                        vmin=vmin,
                        vmax=vmax,
                        cbar=(row == 0 and col == 0),
                        cbar_ax=(cbar_ax if (row == 0 and col == 0) else None),
                        linewidths=0.5,
                        linecolor="gray",
                        annot=True,
                        fmt=".2f",
                        annot_kws={"size": 13, "rotation": 90},
                    )
                    ax.set_title(title, fontsize=35, pad=20)
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.tick_params(axis='y', labelsize=30)
                    ax.tick_params(axis='x', labelsize=10)

                    ax.set_facecolor("yellow")  # Set face color to yellow for all axes

                cbar_ax.set_title("%", fontsize=40)
                cbar_ax.tick_params(labelsize=30)
                cbar_ax.set_box_aspect(20)

                fig.supxlabel("RBP", fontsize=40, x=0.45, y=-0.01)
                fig.supylabel("Position", fontsize=40, x=-0.01)
                plt.suptitle(
                    f"{cell_line}: % Positive/Negative Local SHAP for Bound and NOT Bound Features ('Zero Cutoff' = {cutoff})\
                    \n\nNOTE 0: '{underlying_data}' data mode used to calculate these values\
                    \nNOTE 1: All heatmaps share RBP clustering order from 'Bound Only - % Positive'\
                    \nNOTE 2: Red is for Positive Local SHAP and Blue is for Negative Local SHAP\
                    \nNOTE 3: Colorbar values comparable across all heatmaps (intensity level of blue and red means the same)\
                    \nNOTE 4: Null values indicated by yellow squares\
                    \nNOTE 5: Positive defined as Local SHAP > {cutoff} and Negative defined as Local SHAP < -{cutoff}",
                    fontsize=40, y=1.01
                )
                plt.tight_layout(rect=[0, 0, 0.91, 1])
                plt.show()

        # Second Figure: Scatterplot of % Positive vs % Negative for Bound and NOT Bound, per cell line and cutoff
        for cutoff in CUTOFFS:
            bound_cut = bound_data[cutoff]
            not_bound_cut = not_bound_data[cutoff]
            bound_statuses = [("Bound Only", bound_cut), ("NOT Bound Only", not_bound_cut)]

            fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=300, sharex=True, sharey=True)
            for row_idx, cell_line in enumerate(self.cell_lines):
                for col_idx, (status_label, mode_data) in enumerate(bound_statuses):
                    df_pos = mode_data["positive"][cell_line]
                    df_neg = mode_data["negative"][cell_line]

                    # Ensure index and columns match
                    df_pos = df_pos.sort_index().sort_index(axis=1)
                    df_neg = df_neg.sort_index().sort_index(axis=1)
                    assert df_pos.index.equals(df_neg.index) and df_pos.columns.equals(df_neg.columns), "Index/columns mismatch"

                    # Melt both DataFrames to long format and merge for plotting
                    df_pos_long = df_pos.reset_index().melt(id_vars="index", var_name="RBP", value_name="Percent_Positive")
                    df_neg_long = df_neg.reset_index().melt(id_vars="index", var_name="RBP", value_name="Percent_Negative")
                    plot_df = pd.merge(df_pos_long, df_neg_long, on=["index", "RBP"], how="inner", validate="one_to_one")
                    assert plot_df.shape[0] == df_pos_long.shape[0] == df_neg_long.shape[0], "Mismatch in number of rows after merge"

                    plot_df.rename(columns={"index": "Position"}, inplace=True)

                    # Assert that there are no rows where only one of Percent_Positive or Percent_Negative is null
                    only_one_null = (plot_df["Percent_Positive"].isnull() ^ plot_df["Percent_Negative"].isnull())
                    assert not only_one_null.any(), "There are rows where only one of Percent_Positive or Percent_Negative is null"

                    # Only keep rows where both are not null for plotting
                    plot_df_non_null = plot_df.dropna(subset=["Percent_Positive", "Percent_Negative"])

                    x = plot_df_non_null["Percent_Positive"]
                    y = plot_df_non_null["Percent_Negative"]
                    num_points = len(plot_df_non_null)

                    ax = axes[row_idx, col_idx]
                    sns.scatterplot(
                        x=x,
                        y=y,
                        ax=ax,
                        color="deepskyblue",
                        edgecolor="black",
                        alpha=0.5,
                        s=10
                    )
                    # Label features: average between 1 and 40, or between 40 and 60 on either axis, no duplicates
                    labeled = set()
                    for _, row in plot_df_non_null.iterrows():
                        avg = (row["Percent_Positive"] + row["Percent_Negative"]) / 2
                        label = f"{row['RBP']}_{row['Position']}"
                        should_label = False
                        if 1 < avg < 40:
                            should_label = True
                        elif (40 <= row["Percent_Positive"] <= 60) or (40 <= row["Percent_Negative"] <= 60):
                            should_label = True
                        if should_label and label not in labeled:
                            ax.text(
                                row["Percent_Positive"]-3,
                                row["Percent_Negative"]+2,
                                label,
                                fontsize=4,
                                color="red",
                                alpha=0.8
                            )
                            labeled.add(label)

                    ax.set_title(f"{cell_line} - {status_label}", fontsize=18)
                    ax.set_xlabel("")
                    ax.set_ylabel("")

                    ax.text(
                        0.58, 0.95,
                        f"Points: {num_points}",
                        transform=ax.transAxes,
                        fontsize=12,
                        verticalalignment='center',
                        horizontalalignment='right'
                    )

            plt.suptitle(
                f"% Positive vs % Negative Local SHAP per Feature (cutoff = {cutoff})\n\nREMINDER: 'Bound Only' will have less # points than\n'Not Bound' as some features do not bind\n\nNOTE 1: Using '{underlying_data}' data mode to calculate these values",
                fontsize=18, y=1.01
            )
            fig.supxlabel("% Positive Local SHAP", fontsize=18, y=-0.01)
            fig.supylabel("% Negative Local SHAP", fontsize=18, x=-0.01)
            plt.tight_layout()
            plt.show()

        # Third Figure: Hexbin of % Positive vs % Negative for Bound and NOT Bound, per cell line, per cutoff
        for cutoff in CUTOFFS:
            bound_cut = bound_data[cutoff]
            not_bound_cut = not_bound_data[cutoff]
            bound_statuses = [("Bound Only", bound_cut), ("NOT Bound Only", not_bound_cut)]

            fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300, sharex=True, sharey=True)
            for row_idx, cell_line in enumerate(self.cell_lines):
                for col_idx, (status_label, mode_data) in enumerate(bound_statuses):
                    df_pos = mode_data["positive"][cell_line]
                    df_neg = mode_data["negative"][cell_line]

                    # Ensure index and columns match
                    df_pos = df_pos.sort_index().sort_index(axis=1)
                    df_neg = df_neg.sort_index().sort_index(axis=1)
                    assert df_pos.index.equals(df_neg.index) and df_pos.columns.equals(df_neg.columns), "Index/columns mismatch"

                    # Melt both DataFrames to long format and merge for plotting
                    df_pos_long = df_pos.reset_index().melt(id_vars="index", var_name="RBP", value_name="Percent_Positive")
                    df_neg_long = df_neg.reset_index().melt(id_vars="index", var_name="RBP", value_name="Percent_Negative")
                    plot_df = pd.merge(df_pos_long, df_neg_long, on=["index", "RBP"], how="inner", validate="one_to_one")
                    assert plot_df.shape[0] == df_pos_long.shape[0] == df_neg_long.shape[0], "Mismatch in number of rows after merge"

                    plot_df.rename(columns={"index": "Position"}, inplace=True)

                    # Assert that there are no rows where only one of Percent_Positive or Percent_Negative is null
                    only_one_null = (plot_df["Percent_Positive"].isnull() ^ plot_df["Percent_Negative"].isnull())
                    assert not only_one_null.any(), "There are rows where only one of Percent_Positive or Percent_Negative is null"

                    # Only keep rows where both are not null for plotting
                    plot_df_non_null = plot_df.dropna(subset=["Percent_Positive", "Percent_Negative"])

                    x = plot_df_non_null["Percent_Positive"]
                    y = plot_df_non_null["Percent_Negative"]
                    num_points = len(plot_df_non_null)

                    ax = axes[row_idx, col_idx]
                    hb = ax.hexbin(
                        x, y,
                        gridsize=40,
                        cmap='viridis',
                        mincnt=1,
                        linewidths=1,
                        alpha=1
                    )
                    cbar = plt.colorbar(hb, ax=ax)
                    cbar.ax.set_title("Counts", fontsize=16)

                    ax.set_title(f"{cell_line} - {status_label}", fontsize=20)
                    ax.set_xlabel("")
                    ax.set_ylabel("")

                    ax.text(
                        0.58, 0.95,
                        f"Points: {num_points}",
                        transform=ax.transAxes,
                        fontsize=16,
                        verticalalignment='center',
                        horizontalalignment='right'
                    )

            plt.suptitle(
                f"% Positive vs % Negative Local SHAP per Feature (cutoff = {cutoff})\n\nREMINDER: 'Bound Only' will have less # points than\n'Not Bound' as some features do not bind\n\nNOTE 1: Using '{underlying_data}' data mode to calculate these values",
                fontsize=20, y=1.01
            )
            fig.supxlabel("% Positive Local SHAP", fontsize=20, y=-0.01)
            fig.supylabel("% Negative Local SHAP", fontsize=20, x=-0.01)
            plt.tight_layout()
            plt.show()
        
        # Fourth Figure: Difference heatmap (% Positive - % Negative) for Bound Only, both cell lines in one figure, per cutoff
        for cutoff in CUTOFFS:
            # Prepare difference heatmaps for both cell lines
            diff_bounds = []
            ordered_cols_list = []

            for cell_line in self.cell_lines:
                df_bound_pos = bound_data[cutoff]["positive"][cell_line].sort_index().sort_index(axis=1)
                df_bound_neg = bound_data[cutoff]["negative"][cell_line].sort_index().sort_index(axis=1)
                # Assert index and columns match for subtraction
                assert (df_bound_pos.index.equals(df_bound_neg.index) and df_bound_pos.columns.equals(df_bound_neg.columns)), "Bound: index/columns mismatch"
                
                # Calculate difference heatmap
                diff_bound = df_bound_pos - df_bound_neg
                # Assert values in [-100, 100]
                assert ((diff_bound.values[~np.isnan(diff_bound.values)] >= -100) & (diff_bound.values[~np.isnan(diff_bound.values)] <= 100)).all(), "diff_bound has values outside [-100, 100]"
                
                # Cluster columns using Ward
                linkage = sch.linkage(diff_bound.fillna(0).T, method="ward")
                dendro = sch.dendrogram(linkage, no_plot=True)
                ordered_cols = [diff_bound.columns[i] for i in dendro["leaves"]]
                
                # Reorder heatmap to match clustering
                diff_bound = diff_bound[ordered_cols]
                diff_bounds.append(diff_bound)
                ordered_cols_list.append(ordered_cols)

            # Plot both cell lines in one figure, one subplot per cell line
            fig, axes = plt.subplots(len(self.cell_lines), 1, figsize=(40, 27), dpi=300, sharex=False, sharey=True)
            cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.7])
            vmin, vmax = -100, 100

            for idx, (cell_line, diff_bound) in enumerate(zip(self.cell_lines, diff_bounds)):
                ax = axes[idx]
                sns.heatmap(
                    diff_bound,
                    ax=ax,
                    cmap="bwr",
                    vmin=vmin,
                    vmax=vmax,
                    center=0,
                    cbar=(idx == 0),
                    cbar_ax=(cbar_ax if idx == 0 else None),
                    linewidths=0.5,
                    linecolor="gray",
                    annot=True,
                    fmt=".0f",
                    annot_kws={"size": 20, "rotation": 90},
                )
                
                ax.set_title(f"{cell_line}", fontsize=40, pad=25)
                ax.set_xlabel("")
                ax.set_ylabel("")
                
                ax.tick_params(axis='y', labelsize=36)
                ax.tick_params(axis='x', labelsize=15)
                
                ax.set_facecolor("lightgray")

            cbar_ax.set_title("% Pos - % Neg", fontsize=40, pad=35)
            cbar_ax.tick_params(labelsize=30)
            cbar_ax.set_box_aspect(20)

            fig.supxlabel("RBP", fontsize=50, x=0.47, y=-0.01)
            fig.supylabel("Position", fontsize=50, x=-0.01, y=0.4)
            plt.suptitle(
                f"Bound Only: (% Positive Local SHAP) - (% Negative Local SHAP) (Zero Cutoff = {cutoff})\n\n"
                f"NOTE 0: '{underlying_data}' data mode used to calculate these values\n"
                "NOTE 1: Each cell line's RBPs clustered by Ward\n"
                "NOTE 2: Red = more positive; Blue = more negative; White = 0% Difference\n"
                "NOTE 3: Null values indicated by gray squares\n"
                "NOTE 4: Heatmap colors shared across both cell lines\n"
                "NOTE 5: Zero can have multiple meanings -\n[1] Local SHAP always zero, [2] % positive and % negative cancel each other out\n",
                fontsize=40, y=1.015
            )
            plt.tight_layout(rect=[0, 0, 0.91, 1])

            if cutoff == 0.01: 
                plt.savefig(self.FIGURES["pct_positive_minus_negative_local_SHAP"]["heatmap_bound"], bbox_inches='tight', dpi=600)

            plt.show()
        
        # Fifth Figure: Scatterplot of (Bound: %Pos - %Neg) vs (NOT Bound: %Pos - %Neg) per cell line, for each cutoff
        for cutoff in CUTOFFS:
            for cell_line in self.cell_lines:
                # Prepare difference DataFrames for this cutoff
                diff_bound = bound_data[cutoff]["positive"][cell_line] - bound_data[cutoff]["negative"][cell_line]
                diff_not_bound = not_bound_data[cutoff]["positive"][cell_line] - not_bound_data[cutoff]["negative"][cell_line]

                # Ensure index and columns match
                diff_bound = diff_bound.sort_index().sort_index(axis=1)
                diff_not_bound = diff_not_bound.sort_index().sort_index(axis=1)
                assert diff_bound.index.equals(diff_not_bound.index) and diff_bound.columns.equals(diff_not_bound.columns), "Index/columns mismatch"

                # Melt both DataFrames to long format and merge for plotting
                diff_bound_long = diff_bound.reset_index().melt(id_vars="index", var_name="RBP", value_name="Bound_Diff")
                diff_not_bound_long = diff_not_bound.reset_index().melt(id_vars="index", var_name="RBP", value_name="Not_Bound_Diff")
                plot_df = pd.merge(diff_bound_long, diff_not_bound_long, on=["index", "RBP"], how="inner", validate="one_to_one")
                plot_df.rename(columns={"index": "Position"}, inplace=True)

                # Only keep rows where both are not null for plotting
                plot_df_non_null = plot_df.dropna(subset=["Bound_Diff", "Not_Bound_Diff"])
                x = plot_df_non_null["Bound_Diff"]
                y = plot_df_non_null["Not_Bound_Diff"]

                # Scatterplot
                plt.figure(figsize=(9, 10), dpi=300)
                ax = plt.gca()

                # Set fixed axis limits for all plots
                x0, x1 = -105, 105
                y0, y1 = -105, 105

                # Define quadrant boundaries for activator/repressor
                x_neg = -50
                x_pos = 50
                y_neg = -50
                y_pos = 50

                # Calculate correlations, num points and number/percentage of 
                # [1] points at (0,0), [2] points in repressor quadrant, [3] points in activator quadrant, and [4] remaining points
                num_points = len(plot_df_non_null)
                pearson_corr, _ = pearsonr(x, y)
                spearman_corr, _ = spearmanr(x, y)

                in_repressor = (x < x_neg) & (y > y_pos)
                in_activator = (x > x_pos) & (y < y_neg)
                zero = (x == 0) & (y == 0)

                num_zero_zero = zero.sum()
                num_repressors = in_repressor.sum()
                num_activators = in_activator.sum()
                num_everything_else = (~(in_repressor | in_activator | zero)).sum()

                percent_zero_zero = (num_zero_zero / num_points) * 100 
                percent_repressors = (num_repressors / num_points) * 100
                percent_activators = (num_activators / num_points) * 100
                percent_not_in_either = (num_everything_else / num_points) * 100 

                # Fill the entire plot with lightyellow (uncertain)
                ax.axhspan(y0, y1, xmin=0, xmax=1, facecolor="lightyellow", alpha=0.4, zorder=0)

                # Top left: Repressors (x < -50, y > 50) - lightcoral
                ax.axvspan(x0, x_neg, ymin=(y_pos - y0) / (y1 - y0), ymax=1, facecolor="lightcoral", alpha=0.6, zorder=1)
                ax.axhspan(y_pos, y1, xmin=0, xmax=(x_neg - x0) / (x1 - x0), facecolor="lightcoral", alpha=0.6, zorder=1)

                # Bottom right: Activators (x > 50, y < -50) - lightblue
                ax.axvspan(x_pos, x1, ymin=0, ymax=(y_neg - y0) / (y1 - y0), facecolor="lightblue", alpha=0.6, zorder=1)
                ax.axhspan(y0, y_neg, xmin=(x_pos - x0) / (x1 - x0), xmax=1, facecolor="lightblue", alpha=0.6, zorder=1)

                # Now plot the scatter
                sns.scatterplot(
                    x=x,
                    y=y,
                    color="lightgreen",
                    edgecolor="black",
                    alpha=0.6,
                    s=20,
                    ax=ax
                )

                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)

                ax.set_xlabel('')
                ax.set_ylabel('')

                # Plot y=-x line
                min_val = min(x.min(), y.min())
                max_val = max(x.max(), y.max())
                ax.plot([min_val, max_val], [-min_val, -max_val], color="chocolate", linestyle="--", linewidth=1, label="y=-x")

                # Annotate only points that are not zero and not in the top left or bottom right corners
                threshold = 80  # adjust as needed for your data scale
                mask = (
                    ~((plot_df_non_null["Bound_Diff"] == 0) & (plot_df_non_null["Not_Bound_Diff"] == 0)) &
                    ~(
                        ((plot_df_non_null["Bound_Diff"] < -threshold) & (plot_df_non_null["Not_Bound_Diff"] > threshold)) |  # top left
                        ((plot_df_non_null["Bound_Diff"] > threshold) & (plot_df_non_null["Not_Bound_Diff"] < -threshold))    # bottom right
                    )
                )
                for _, row in plot_df_non_null[mask].iterrows():
                    label = f"{row['RBP']}_{row['Position']}"
                    ax.text(
                        row["Bound_Diff"]-4,
                        row["Not_Bound_Diff"]+1.2,
                        label,
                        fontsize=5,
                        color="navy",
                        alpha=0.8
                    )

                # Add quadrant labels
                # Top left: Repressors
                ax.text(
                    x0 + 0.05 * (x1 - x0),
                    y1 + 0.01 * (y1 - y0),
                    "Repressors",
                    color="firebrick",
                    fontsize=16,
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                    alpha=0.8
                )
                # Top right: Unclear
                ax.text(
                    x1 - 0.18 * (x1 - x0),
                    y1 + 0.01 * (y1 - y0),
                    "Unclear",
                    color="goldenrod",
                    fontsize=16,
                    fontweight="bold",
                    ha="right",
                    va="bottom",
                    alpha=0.8
                )
                # Bottom left: Unclear
                ax.text(
                    x0 + 0.18 * (x1 - x0),
                    y0 - 0.01 * (y1 - y0),
                    "Unclear",
                    color="goldenrod",
                    fontsize=16,
                    fontweight="bold",
                    ha="left",
                    va="top",
                    alpha=0.8
                )
                # Bottom right: Activators
                ax.text(
                    x1 - 0.04 * (x1 - x0),
                    y0 - 0.01 * (y1 - y0),
                    "Activators",
                    color="royalblue",
                    fontsize=16,
                    fontweight="bold",
                    ha="right",
                    va="top",
                    alpha=0.8
                )

                ax.set_title(f"{cell_line}: Bound[%Pos - %Neg] vs NOT Bound[%Pos - %Neg] (cutoff={cutoff})\n\nREMINDER: 'never bound' features are excluded (null values)\n\nNOTE: '{underlying_data}' data mode used for calculating all values", fontsize=14, y=1.09)

                ax.text(
                    0.98, 0.7,
                    f"Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}\nPoints: {num_points}\n(0,0): {num_zero_zero} - {percent_zero_zero:.2f}%\nRepressors: {num_repressors} - {percent_repressors:.2f}%\nActivators: {num_activators} - {percent_activators:.2f}%\nEverything else: {num_everything_else} - {percent_not_in_either:.2f}%",
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='bottom',
                    horizontalalignment='right'
                )

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_position('zero')
                ax.spines['bottom'].set_position('zero')

                ax.text(0.5, -0.1, "Bound Only: % Positive - % Negative", fontsize=12, ha='center', va='top', transform=ax.transAxes)
                ax.text(-0.1, 0.5, "NOT Bound Only: % Positive - % Negative", fontsize=12, ha='right', va='center', rotation=90, transform=ax.transAxes)
                plt.tight_layout()
                plt.show()

        # Sixth Figure: Scatterplot for matching RBPs between both cell lines
        for cutoff in CUTOFFS:
            # Get matching RBPs between both cell lines for this cutoff
            bound_cut = bound_data[cutoff]
            rbps_0 = set(bound_cut["positive"][self.cell_lines[0]].columns)
            rbps_1 = set(bound_cut["positive"][self.cell_lines[1]].columns)

            matching_rbps = sorted(rbps_0 & rbps_1)
            assert len(matching_rbps) > 0, f"There should be matching RBPs between {self.cell_lines[0]} and {self.cell_lines[1]} for cutoff {cutoff}"

            for cell_line in self.cell_lines:
                df_pos = bound_cut["positive"][cell_line][matching_rbps]
                df_neg = bound_cut["negative"][cell_line][matching_rbps]
                
                # Ensure index and columns match
                df_pos = df_pos.sort_index().sort_index(axis=1)
                df_neg = df_neg.sort_index().sort_index(axis=1)
                assert df_pos.index.equals(df_neg.index) and df_pos.columns.equals(df_neg.columns), "Index/columns mismatch"

                # Melt to long format and compute %pos - %neg, aligning by feature name
                df_pos_long = df_pos.reset_index().melt(id_vars="index", var_name="RBP", value_name="Percent_Positive")
                df_pos_long["Feature"] = df_pos_long["RBP"].astype(str) + "_" + df_pos_long["index"].astype(str)

                df_neg_long = df_neg.reset_index().melt(id_vars="index", var_name="RBP", value_name="Percent_Negative")
                df_neg_long["Feature"] = df_neg_long["RBP"].astype(str) + "_" + df_neg_long["index"].astype(str)

                # Merge on Feature to ensure alignment
                df_long = pd.merge(
                    df_pos_long[["Feature", "Percent_Positive"]],
                    df_neg_long[["Feature", "Percent_Negative"]],
                    on="Feature",
                    how="inner"
                )
                
                # Extract RBP and index (Position) back if needed
                df_long["RBP"] = df_long["Feature"].str.rsplit("_", n=1).str[0]
                df_long["index"] = df_long["Feature"].str.rsplit("_", n=1).str[1]
                df_long["Pct_Pos_Minus_Neg"] = df_long["Percent_Positive"] - df_long["Percent_Negative"]
                df_long["Cell Line"] = cell_line

                if cell_line == self.cell_lines[0]:
                    df_long_0 = df_long
                else:
                    df_long_1 = df_long            

            # Merge on Feature to get matching features
            merged = pd.merge(
                df_long_0[["Feature", "Pct_Pos_Minus_Neg"]],
                df_long_1[["Feature", "Pct_Pos_Minus_Neg"]],
                on="Feature",
                suffixes=(f"_{self.cell_lines[0]}", f"_{self.cell_lines[1]}")
            )

            # Remove rows with NaN in either column
            merged = merged.dropna(axis=0, how='any')
            x = merged[f"Pct_Pos_Minus_Neg_{self.cell_lines[0]}"]
            y = merged[f"Pct_Pos_Minus_Neg_{self.cell_lines[1]}"]

            pearson_corr, _ = pearsonr(x, y)
            spearman_corr, _ = spearmanr(x, y)
            num_points = len(merged)

            fig, ax = plt.subplots(figsize=(6, 5), dpi=200)

            sns.scatterplot(x=x, y=y, color="deepskyblue", edgecolor="black", alpha=0.1, s=20, ax=ax)
            # Make y=x line thinner
            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())
            ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=0.7, label="y=x")

            # Move axes to cross at (0,0)
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            
            # Set x and y limits to -200 to 200 for space, but only show tick labels for -100 to 100
            ax.set_xlim(-150, 150)
            ax.set_ylim(-150, 150)

            # Only show ticks and labels for -100 to 100
            xticks = [tick for tick in ax.get_xticks() if -100 <= tick <= 100]
            yticks = [tick for tick in ax.get_yticks() if -100 <= tick <= 100]

            ax.set_xticks(xticks)
            ax.set_yticks(yticks)

            ax.set_xticklabels([str(int(tick)) for tick in xticks])
            ax.set_yticklabels([str(int(tick)) for tick in yticks])

            # Place x and y axis labels outside the plot using transAxes for precise positioning
            ax.text(
                0.5, -0.04,
                f"{self.cell_lines[0]}: %Pos - %Neg",
                fontsize=12,
                ha='center',
                va='top',
                transform=ax.transAxes
            )
            ax.text(
                -0.04, 0.5,
                f"{self.cell_lines[1]}: %Pos - %Neg",
                fontsize=12,
                ha='right',
                va='center',
                rotation=90,
                transform=ax.transAxes
            )

            ax.set_xlabel("")
            ax.set_ylabel("")

            # Set the title at y=1.05
            ax.set_title(f"Matching RBPs: Bound-Only % Positive - % Negative (cutoff={cutoff})", fontsize=8, y=1.05)

            # Move correlation/points text outside the plot, center right
            fig.text(
                0.15, 0.75,
                f"Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}\nPoints: {num_points}",
                fontsize=8,
                verticalalignment='center',
                horizontalalignment='left'
            )

            # Annotate points where both x and y < -50 or both x and y > 50, using adjustText for non-overlapping labels
            texts = []
            for idx, row in merged.iterrows():
                x_val = row[f"Pct_Pos_Minus_Neg_{self.cell_lines[0]}"]
                y_val = row[f"Pct_Pos_Minus_Neg_{self.cell_lines[1]}"]
                if (x_val < -50 and y_val < -50) or (x_val > 50 and y_val > 50):
                    texts.append(
                        ax.text(
                            x_val,
                            y_val,
                            row["Feature"],
                            fontsize=5,  # Reduced fontsize
                            color="black",
                            alpha=0.8
                        )
                    )
            
            adjustText.adjust_text(
                texts,
                ax=ax,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                ensure_inside_axes=False,
                expand_text=(2.0, 2.0),   # Increase distance between text labels
                expand_points=(2.0, 2.0),  # Increase distance from points
                force_text=(0.5, 0.5),     # Increase repulsion between texts
                force_points=(0.5, 0.5),   # Increase repulsion between texts and points
                lim=1000                    # Allow more iterations for better spacing
            )

            if cutoff == 0.01:
                plt.savefig(
                    self.FIGURES["pct_positive_minus_negative_local_SHAP"]["matching_features_bound_scatter"], 
                    bbox_inches='tight',
                    dpi=600
                )

            plt.show()


    def parallel_helper_local_SHAP_percent_non_zero(self, data, zero_cutoff): 

        assert len(data) > 0, "Data must not be empty"
        assert data.name.endswith("_shap"), "Data must be a local SHAP Series"
        assert type(data) is pl.Series, "Data must be a polars Series"

        return (
            (data.abs() > zero_cutoff).sum() / len(data)
        ) * 100
    

    def calculate_local_SHAP_percent_non_zero(self, mode=None):
        assert mode == "Unique-Binding", "Mode must be 'Unique-Binding' for this function"

        ZERO_CUTOFFS = [1e-15, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
        OUTPUT_FILE = self.CACHE_INFO["local_SHAP_percent_non_zero"][mode]

        if os.path.exists(OUTPUT_FILE):
            logger.success(f"FROM CACHE: Loading percent non-zero local SHAP for mode {mode} from {OUTPUT_FILE}")
            return pd.read_csv(OUTPUT_FILE, sep="\t")

        else:
            logger.info(f"Calculating percent non-zero local SHAP for mode: {mode}")

            lfs = self.load_final_SHAP_data(underlying_data="Unique-Binding", as_lazyframe=True)

            all_results = []
            for cell_line in self.cell_lines:

                logger.info(f"Calculating percent non-zero local SHAP for cell line: {cell_line}")

                schema = lfs[cell_line].collect_schema().names()
                shap_cols = [col for col in schema if col.endswith("_shap")]

                mean_df = lfs[cell_line].select(shap_cols).collect()
                zero_cutoffs = ZERO_CUTOFFS.copy() # for clarity

                # Prepare arguments for parallel execution: (feature_series, zero_cutoff)
                tasks = []
                for feature in mean_df.columns:
                    series = mean_df[feature]

                    for zero_cutoff in zero_cutoffs:
                        tasks.append((series, zero_cutoff, feature))

                # Use ProcessPoolExecutor for parallel computation
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.get_slurm_job_num_cpus()) as executor:
                    future_to_args = {
                        executor.submit(self.parallel_helper_local_SHAP_percent_non_zero, series, zero_cutoff): (feature, zero_cutoff)
                        for series, zero_cutoff, feature in tasks
                    }
                    for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_args), total=len(future_to_args), desc=f"{cell_line} percent non-zero"):
                        feature, zero_cutoff = future_to_args[future]
                        percent_zero = future.result()
                        all_results.append({
                            "Cell Line": cell_line,
                            "Data Mode": mode,
                            "Feature": feature,
                            "Zero Cutoff": zero_cutoff,
                            "% Non Zero": percent_zero
                        })

            # Convert all_results to DataFrame and save to output file
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(OUTPUT_FILE, sep="\t", index=False)
            logger.success(f"Saved percent non zero local SHAP for all cell lines in mode {mode} to {OUTPUT_FILE}")

            return results_df
    

    def plot_local_SHAP_percent_non_zero(self): 

        # Load percent zero local SHAP data for both modes
        unique_binding_df = self.calculate_local_SHAP_percent_non_zero(mode="Unique-Binding")

        # Add a "Position" column by extracting the position using get_RBP_position
        
        rbp_pos = unique_binding_df["Feature"].apply(lambda x: self.get_RBP_position(x))
        unique_binding_df["RBP"] = rbp_pos.apply(lambda x: x[0])
        unique_binding_df["Position"] = rbp_pos.apply(lambda x: x[1])

        # Prepare zero cutoff order for x-axis
        zero_cutoff_order = sorted(unique_binding_df["Zero Cutoff"].unique())
        # Prepare position order for hue
        position_order = sorted(unique_binding_df["Position"].unique())

        violinplot_configs = [
            {
                "use_hue": False,
                "palette": sns.color_palette("viridis", n_colors=len(zero_cutoff_order)),
                "hue": None,
                "legend": False
            },
            {
                "use_hue": True,
                "palette": sns.color_palette("tab10", n_colors=len(position_order)),
                "hue": "Position",
                "legend": True
            }
        ]

        for config in violinplot_configs:
            fig, axes = plt.subplots(len(self.cell_lines), 1, figsize=(20, 12), dpi=300, sharex=True, sharey=True)
            legend_handles = None
            legend_labels = None

            for cell_idx, cell_line in enumerate(self.cell_lines):
                ax = axes[cell_idx]
                plot_df = unique_binding_df[unique_binding_df["Cell Line"] == cell_line].copy()
                plot_df["Zero Cutoff"] = plot_df["Zero Cutoff"].astype(float).sort_values(ascending=True)

                if not config["use_hue"]:
                    sns.violinplot(
                        data=plot_df,
                        x="Zero Cutoff",
                        y="% Non Zero",
                        order=zero_cutoff_order,
                        palette=config["palette"],
                        ax=ax,
                        cut=0,
                        linewidth=1,
                        density_norm="width"
                    )
                else:
                    sns.violinplot(
                        data=plot_df,
                        x="Zero Cutoff",
                        y="% Non Zero",
                        hue=config["hue"],
                        order=zero_cutoff_order,
                        hue_order=position_order,
                        palette=config["palette"],
                        ax=ax,
                        cut=0,
                        linewidth=1,
                        density_norm="width",
                        split=False,
                        inner="box",
                        width=0.7
                    )
                    # Only collect legend handles/labels from the first plot
                    if legend_handles is None and config["legend"]:
                        handles, labels = ax.get_legend_handles_labels()
                        legend_handles, legend_labels = handles, labels


                ax.set_title(f"{cell_line} - Unique-Binding", fontsize=30)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_ylim(-10, 110)
                # ax.set_xscale("log")
                ax.tick_params(axis="x", labelrotation=45, labelsize=20, labelbottom=True)
                ax.tick_params(axis="y", labelsize=18)

            # Add a single legend to the right if using hue
            if config["use_hue"]:
                fig.legend(
                    legend_handles, legend_labels, title="Position",
                    bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=30, title_fontsize=30
                )

            if config["hue"] == "Position":
                suffix = "By Position"
            else:
                suffix = ""

            fig.suptitle(f"% Local SHAP Non-Zero per Cutoff {suffix}\nNOTE: x-axis is the exact same across all plots", fontsize=30, y=1.01)
            fig.supxlabel("Zero Cutoff", fontsize=28, y=-0.01)
            fig.supylabel("% Non Zero", fontsize=28, x=-0.01)

            fig.tight_layout(rect=[0, 0, 1, 1])
            plt.show()

        # Second Figure: Line plot of ALL FEATURES % Non-Zero Cutoff
        fig, axes = plt.subplots(len(self.cell_lines), 1, figsize=(18, 12), dpi=300, sharex=True, sharey=True)

        for row_idx, cell_line in enumerate(self.cell_lines):
            ax = axes[row_idx]
            # Subset to this cell line and mode
            plot_df = unique_binding_df[unique_binding_df["Cell Line"] == cell_line]
            # Iterate by each unique feature and plot its line
            for feature in plot_df["Feature"].unique():
                feature_df = plot_df[plot_df["Feature"] == feature]
                sns.lineplot(
                    data=feature_df,
                    x="Zero Cutoff",
                    y="% Non Zero",

                    ax=ax,
                    label=None,
                    linewidth=0.7,
                    alpha=0.7,
                    marker='o',   # Show each point in the line
                    markersize=6
                )
            ax.set_title(f"{cell_line} - Unique-Binding", fontsize=20)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_ylim(-10, 110)
            ax.set_xscale("log")
            ax.tick_params(axis="x", labelsize=16)
            ax.tick_params(axis="y", labelsize=16)

        fig.suptitle("Percent Non-Zero Local SHAP per Feature\nNOTE: each feature is a line", fontsize=24)
        fig.supxlabel("Zero Cutoff", fontsize=20, y=-0.01)
        fig.supylabel("% Non Zero", fontsize=20, x=-0.01)

        fig.tight_layout()
        fig.show()

        # Third Figure: Line plot of % Non-Zero vs Zero Cutoff across all features and cell lines
        plt.figure(figsize=(6,5), dpi=300)

        sns.lineplot(
            data=unique_binding_df,
            x="Zero Cutoff",
            y="% Non Zero",
            hue="Cell Line",
            markers=True,
            dashes=True,
            n_boot = 5000,
            seed = 17
        )
        plt.xscale("log")
        plt.xlabel("Zero Cutoff", fontsize=10)
        plt.ylabel("% Non Zero", fontsize=10)
        plt.title("% Local SHAP Non-Zero vs Zero Cutoff\nNOTE: 'Confidence Interval' bands included around each line", fontsize=10)
        plt.legend(title="Cell Line", fontsize=8, loc="lower left")

        plt.show()

        # Fourth Figure: Line plot of % Non-Zero vs Zero Cutoff by Position per cell line
        fig, axes = plt.subplots(len(self.cell_lines), 1, figsize=(10, 8), dpi=300, sharex=True, sharey=True)

        # Use a highly contrasting color palette for 6 positions
        palette = sns.color_palette("tab10", n_colors=6)

        for ax, cell_line in zip(axes, self.cell_lines):
            plot_df = unique_binding_df[unique_binding_df["Cell Line"] == cell_line].copy()
            plot_df["Zero Cutoff"] = plot_df["Zero Cutoff"].astype(float)
            # Plot with seaborn lineplot: hue=Position
            sns.lineplot(
                data=plot_df,
                x="Zero Cutoff",
                y="% Non Zero",
                hue="Position",
                markers=True,
                dashes=True,
                hue_order=position_order,
                palette=palette,
                ax=ax,
                errorbar=None,  
            )
            ax.set_xscale("log")
            ax.set_title(cell_line, fontsize=14)

            ax.set_xlabel("")
            ax.set_ylabel("")

            ax.tick_params(axis="x", labelsize=14)
            ax.tick_params(axis="y", labelsize=14)

        # Remove legends from both axes
        for ax in axes:
            ax.get_legend().remove()

        # Create a single legend on the right
        handles, labels = axes[0].get_legend_handles_labels()
        # Build legend for hue (Position)
        hue_handles = []
        hue_labels = []
        for h, l in zip(handles, labels):
            if l in map(str, position_order):
                hue_handles.append(h)
                hue_labels.append(l)
        # Place legend for hue (Position)
        legend1 = Legend(fig, hue_handles, hue_labels, title="Position", loc="center left", bbox_to_anchor=(1.01, 0.55), fontsize=14, title_fontsize=16)
        fig.add_artist(legend1)

        plt.suptitle("% Local SHAP Non-Zero vs Zero Cutoff\nNOTE: each line is a position\nNOTE 2: Error around each line ('Confidence Interval') disabled for clarity ", fontsize=20, y=1.01, x= 0.6)

        fig.supxlabel("Zero Cutoff", fontsize=16, y=-0.01)
        fig.supylabel("% Non Zero", fontsize=16, x=-0.01)

        plt.tight_layout(rect=[0, 0, 0.99, 1])
        plt.show()


    def calculate_activator_repressor_behavior_score(self, binding_mode=None, underlying_data=None):
        assert binding_mode in ["Bound-Only", "NOT-Bound-Only"], "binding_mode must be 'Bound-Only' or 'NOT-Bound-Only' for this function"
        assert underlying_data == "Unique-Binding", "underlying_data must be 'Unique-Binding' for this function"

        OUTPUT_FILE = self.CACHE_INFO["activator_repressor_behavior_score"][binding_mode]

        if os.path.exists(OUTPUT_FILE):
            logger.success(f"FROM CACHE: Activator/Repressor behavior score already exists for binding_mode={binding_mode}")
            return pd.read_csv(OUTPUT_FILE, sep="\t")

        else: 
            ZERO_CUTOFF = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

            logger.info(f"Calculating Activator/Repressor behavior score for binding_mode={binding_mode}")

            local_shap_gen = self.get_local_SHAP_based_on_binding_and_covariates(
                binding_value=1 if binding_mode == "Bound-Only" else 0,
                binding_pattern_type=underlying_data
            )

            results = []
            for cell_line, feature, series in tqdm.tqdm(local_shap_gen, desc=f"Computing ARBS/NARBS for {binding_mode}"):
                arr = series.to_numpy()
                for zero_cutoff in ZERO_CUTOFF:
                    pos_mask = arr > zero_cutoff
                    neg_mask = arr < -(zero_cutoff)

                    num_pos = np.sum(pos_mask)
                    sum_pos = np.sum(arr[pos_mask]) if num_pos > 0 else 0.0
                    num_neg = np.sum(neg_mask)
                    sum_neg = np.sum(arr[neg_mask]) if num_neg > 0 else 0.0

                    if all(x == 0 for x in [num_pos, num_neg, sum_pos, sum_neg]):
                        arbs = np.nan
                        narbs = np.nan
                        
                    else:
                        numerator = abs(sum_pos) - abs(sum_neg)
                        # ARBS: not normalized metric
                        arbs = numerator
                        # NARBS: normalized metric (|sum_pos| - |sum_neg|) / (|sum_pos| + |sum_neg|)
                        denominator = abs(sum_pos) + abs(sum_neg)
                        narbs = numerator / denominator if denominator != 0 else np.nan

                    num_non_zeros = np.sum(np.abs(arr) > zero_cutoff)
                    percent_non_zeros = (num_non_zeros / len(arr)) * 100

                    results.append({
                        "Data Mode": "Unique Binding",
                        "Binding Type": binding_mode.replace("-", " "),
                        "Cell Line": cell_line,
                        "Feature": feature.replace("_shap", ""),
                        "Zero Cutoff": zero_cutoff,
                        "# Non-Zeros": num_non_zeros,
                        "% Non-Zeros": percent_non_zeros,
                        "Num Positive": num_pos,
                        "Sum Positive": sum_pos,
                        "Num Negative": num_neg,
                        "Sum Negative": sum_neg,
                        "ARBS": arbs,
                        "NARBS": narbs,
                    })

            df = pd.DataFrame(results)
            df = df.sort_values(by=["Cell Line", "Feature", "Zero Cutoff"]).reset_index(drop=True)
            df.to_csv(OUTPUT_FILE, sep="\t", index=False)

            logger.success(f"Saved Activator/Repressor behavior score for {binding_mode} to {OUTPUT_FILE}")

    
    def plot_activator_repressor_behavior_score(self, underlying_data=None):
        assert underlying_data in ["Unique-Binding"], "underlying_data must be 'Unique-Binding' for this function"

        # Load ARBS/NARBS scores for Bound-Only and NOT-Bound-Only from cache
        bound_df = self.calculate_activator_repressor_behavior_score(binding_mode="Bound-Only", underlying_data=underlying_data)
        not_bound_df = self.calculate_activator_repressor_behavior_score(binding_mode="NOT-Bound-Only", underlying_data=underlying_data)

        # Assert that all real number values for "NARBS" are between -1 and 1 (inclusive)
        for df in [bound_df, not_bound_df]:
            narbs_real = df["NARBS"].dropna()
            assert ((narbs_real >= -1) & (narbs_real <= 1)).all(), "NARBS values out of range [-1, 1]"

        # Define log_modulus once for consistent use
        def log_modulus(x):
            return np.sign(x) * np.log10(np.abs(x)) if x != 0 else np.nan

        for score_col in ["ARBS", "NARBS"]:
            for zero_cutoff in sorted(bound_df["Zero Cutoff"].unique())[:-1]: 
                for cell_line in self.cell_lines:
                    # Subset to this cell line and cutoff
                    bound_sub = bound_df[(bound_df["Cell Line"] == cell_line) & (bound_df["Zero Cutoff"] == zero_cutoff)].copy(deep=True)
                    not_bound_sub = not_bound_df[(not_bound_df["Cell Line"] == cell_line) & (not_bound_df["Zero Cutoff"] == zero_cutoff)].copy(deep=True)

                    # Split Feature into RBP and Position
                    for df in [bound_sub, not_bound_sub]:
                        df[["RBP", "Position"]] = df["Feature"].str.rsplit("_", n=1, expand=True)
                        df["Position"] = df["Position"].astype(int)

                        # Assert unique RBP/Position
                        assert not df.duplicated(subset=["RBP", "Position"]).any(), "Duplicate RBP/Position found"

                    # Pivot to heatmap
                    bound_heatmap = bound_sub.pivot(index="Position", columns="RBP", values=score_col)
                    not_bound_heatmap = not_bound_sub.pivot(index="Position", columns="RBP", values=score_col)

                    # If ARBS, apply log-modulus transformation and set zeros to NaN
                    if score_col == "ARBS":
                        bound_heatmap = bound_heatmap.map(log_modulus)
                        not_bound_heatmap = not_bound_heatmap.map(log_modulus)

                    # Cluster columns of bound_heatmap using Ward
                    linkage = sch.linkage(bound_heatmap.fillna(0).T, method="ward")
                    dendro = sch.dendrogram(linkage, no_plot=True)
                    ordered_cols = [bound_heatmap.columns[i] for i in dendro["leaves"]]

                    # Reorder both heatmaps to match clustering
                    bound_heatmap = bound_heatmap[ordered_cols]
                    not_bound_heatmap = not_bound_heatmap[ordered_cols]

                    # Set vmin/vmax for shared colorbar
                    all_vals = pd.concat([bound_heatmap.stack(), not_bound_heatmap.stack()])
                    vmin = np.nanmin(all_vals)
                    vmax = np.nanmax(all_vals)

                    # Plot
                    fig, axes = plt.subplots(2, 1, figsize=(36, 20), dpi=300, sharex=True, sharey=True)
                    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

                    for idx, (data, title) in enumerate([
                        (bound_heatmap, "Bound Only (Ward clustered)"),
                        (not_bound_heatmap, "NOT Bound Only (RBP order matches above)"),
                    ]):
                        ax = axes[idx]
                        sns.heatmap(
                            data,
                            ax=ax,
                            cmap="bwr",
                            vmin=vmin,
                            vmax=vmax,
                            center=0,
                            cbar=(idx == 0),
                            cbar_ax=(cbar_ax if idx == 0 else None),
                            linewidths=0.5,
                            linecolor="gray",
                            annot=True,
                            fmt=".2f",
                            annot_kws={"size": 14, "rotation": 90},
                        )
                        ax.set_title(title, fontsize=28, pad=15)
                        ax.set_xlabel("")
                        ax.set_ylabel("")
                        ax.tick_params(axis='y', labelsize=22)
                        ax.tick_params(axis='x', labelsize=14)
                        ax.set_facecolor("lightyellow")

                    cbar_ax.set_title(
                        f"{score_col}\n(log-modulus)" if score_col == "ARBS" else score_col,
                        fontsize=28, pad=20
                    )
                    cbar_ax.tick_params(labelsize=28)
                    cbar_ax.set_box_aspect(20)

                    # Reinstate original suptitle, supxlabel, and supylabel
                    fig.supxlabel("RBP", fontsize=40, x=0.45)
                    fig.supylabel("Position", fontsize=40, x=-0.01)
                    plt.suptitle(
                        f"{cell_line}: {score_col} Heatmaps (Zero Cutoff={zero_cutoff})\n\nNOTE 1: Bound Only clustered by Ward, NOT Bound Only matches RBP order\nNOTE 2: All metric values come from {underlying_data}\nNOTE 3: Null values shown in yellow squares\nNOTE 4: Heatmap values share a common colorbar"
                        + ("\nNOTE 5: ARBS values are log-modulus transformed (zeros set to NaN)" if score_col == "ARBS" else ""),
                        fontsize=40, y=1.01
                    )
                    plt.tight_layout(rect=[0, 0, 0.91, 1])
                    plt.show()


                    if score_col == "NARBS": 
                        # Prepare data for scatterplot: compare Bound-Only vs NOT-Bound-Only for each feature
                        # Merge bound_sub and not_bound_sub on Feature
                        merged = pd.merge(
                            bound_sub[["Feature", score_col]],
                            not_bound_sub[["Feature", score_col]],
                            on="Feature",
                            suffixes=("_bound", "_not_bound")
                        )
                        
                        # Drop rows where either value is NaN
                        merged = merged.dropna(subset=[f"{score_col}_bound", f"{score_col}_not_bound"])

                        x = merged[f"{score_col}_bound"]
                        y = merged[f"{score_col}_not_bound"]

                        # Calculate correlations and number of points
                        pearson_corr, _ = pearsonr(x, y)
                        spearman_corr, _ = spearmanr(x, y)
                        num_points = len(merged)

                        # Calculate category counts and percentages
                        activator_mask = (x < -0.5) & (y > 0.5)
                        repressor_mask = (x > 0.5) & (y < -0.5)
                        unclear_mask = ~(activator_mask | repressor_mask)

                        num_activator = activator_mask.sum()
                        num_repressor = repressor_mask.sum()
                        num_unclear = unclear_mask.sum()

                        percent_activator = (num_activator / num_points) * 100 
                        percent_repressor = (num_repressor / num_points) * 100 
                        percent_unclear = (num_unclear / num_points) * 100

                        # Scatterplot
                        plt.figure(figsize=(5,6), dpi=300)
                        ax = plt.gca()

                        # Explicitly set axis limits to [-1.1, 1.1]
                        x0, x1 = -1.05, 1.05
                        y0, y1 = -1.05, 1.05

                        # Fill the entire region from -1.1 to 1.1 on both axes with lightyellow (Unclear)
                        ax.add_patch(plt.Rectangle(
                            (x0, y0), x1 - x0, y1 - y0,
                            facecolor="#fff9db", alpha=0.7, zorder=0
                        ))

                        # Repressor quadrant: x < -0.5, y > 0.5 (mediumblue, top left)
                        ax.add_patch(plt.Rectangle(
                            (x0, 0.5), -0.5 - x0, y1 - 0.5,
                            facecolor="mediumblue", alpha=0.18, zorder=1, linewidth=0
                        ))

                        # Activator quadrant: x > 0.5, y < -0.5 (mediumred, bottom right)
                        ax.add_patch(plt.Rectangle(
                            (0.5, y0), x1 - 0.5, -0.5 - y0,
                            facecolor="#cd2626", alpha=0.18, zorder=1, linewidth=0  # mediumred hex
                        ))

                        # Add gold negative diagonal from top left to bottom right
                        ax.plot([x0, x1], [y1, y0], color="gold", linestyle="--", linewidth=1, zorder=1)

                        # Scatterplot with lightgreen dots
                        sns.scatterplot(x=x, y=y, color="lightgreen", edgecolor="black", alpha=0.6, s=10, ax=ax)

                        ax.set_xlabel("")
                        ax.set_ylabel("")

                        # Add x and y labels using transAxes for precise placement
                        ax.text(
                            0.5, -0.06,  # x, y in axes fraction
                            f"Bound {score_col}",
                            fontsize=8,
                            ha='center',
                            va='top',
                            transform=ax.transAxes
                        )
                        ax.text(
                            -0.06, 0.5,
                            f"NOT Bound {score_col}",
                            fontsize=8,
                            ha='right',
                            va='center',
                            rotation=90,
                            transform=ax.transAxes
                        )

                        # Label all points that are not (x > 0.9 and y < -0.9) AND not (x < -0.9 and y > 0.9)
                        for _, row in merged.iterrows():
                            xval = row[f"{score_col}_bound"]
                            yval = row[f"{score_col}_not_bound"]
                            # Label if NOT in (x > 0.9 and y < -0.9) AND NOT in (x < -0.9 and y > 0.9)
                            if not ((xval > 0.9 and yval < -0.9) or (xval < -0.9 and yval > 0.9)):
                                ax.text(
                                    xval-.03,
                                    yval+.015,
                                    row["Feature"],
                                    fontsize=3,
                                    color="navy",
                                    alpha=0.7
                                )

                        # Set axis limits explicitly (expanded to make space for labels)
                        ax.set_xlim(x0, x1)
                        ax.set_ylim(y0, y1)

                        # Add quadrant labels
                        # Repressor (top left)
                        ax.text(
                            x0 + 0.05 * (x1 - x0),
                            y1 + 0.01 * (y1 - y0),
                            "Repressor",
                            color="mediumblue",
                            fontsize=10,
                            fontweight="bold",
                            ha="left",
                            va="bottom",
                            alpha=0.8
                        )
                        # Activator (bottom right)
                        ax.text(
                            x1 - 0.04 * (x1 - x0),
                            y0 - 0.01 * (y1 - y0),
                            "Activator",
                            color="#cd2626",
                            fontsize=10,
                            fontweight="bold",
                            ha="right",
                            va="top",
                            alpha=0.8
                        )
                        # Unclear (middle of negative x and negative y axes)
                        ax.text(
                            0.5 * (x0 + 0),
                            0.5 * (y0 + 0),
                            "Unclear",
                            color="goldenrod",
                            fontsize=10,
                            fontweight="bold",
                            ha="center",
                            va="center",
                            alpha=0.7
                        )
                        # Unclear (middle of positive x and positive y axes)
                        ax.text(
                            0.5 * (0 + x1),
                            0.5 * (0 + y1),
                            "Unclear",
                            color="goldenrod",
                            fontsize=10,
                            fontweight="bold",
                            ha="center",
                            va="center",
                            alpha=0.7
                        )

                        ax.text(
                            0.98, 0.52,
                            f"Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}\n\n# Points: {num_points}\n"
                            f"Activator: {num_activator} ({percent_activator:.1f}%)\n"
                            f"Repressor: {num_repressor} ({percent_repressor:.1f}%)\n"
                            f"Unclear: {num_unclear} ({percent_unclear:.1f}%)",
                            transform=ax.transAxes,
                            fontsize=6,
                            verticalalignment='bottom',
                            horizontalalignment='right',
                        )

                        # Build the title string
                        title = (
                            f"{cell_line}: Bound vs NOT Bound {score_col} (Zero Cutoff={zero_cutoff})\n\n"
                            f"NOTE 1: Values calculated from {underlying_data} data\n"
                            f"NOTE 2: Features w/ null values for either axis were dropped\n"
                        )
                        if score_col == "ARBS":
                            title += "NOTE 3: ARBS values are log-modulus transformed (zeros set to NaN)\n\n"
                        ax.set_title(title, fontsize=8, y=1.02)

                        # Move axes to the center
                        ax.spines['left'].set_position('zero')
                        ax.spines['bottom'].set_position('zero')
                        ax.spines['right'].set_color('none')
                        ax.spines['top'].set_color('none')
                        ax.xaxis.set_ticks_position('bottom')
                        ax.yaxis.set_ticks_position('left')
                        ax.tick_params(axis="x", labelsize=6)
                        ax.tick_params(axis="y", labelsize=6)

                        plt.tight_layout()
                        plt.show()


            # Add Position column to both DataFrames
            for df in [bound_df, not_bound_df]:
                df["Position"] = df["Feature"].str.rsplit("_", n=1, expand=True)[1].astype(int)
                df["RBP"] = df["Feature"].str.rsplit("_", n=1, expand=True)[0]

            # Get all unique positions for style order
            all_positions = sorted(
                pd.concat([bound_df["Position"], not_bound_df["Position"]]).unique()
            )

            # If plotting ARBS, apply log modulus transformation and set zeros to NaN
            plot_bound_df = bound_df.copy()
            plot_not_bound_df = not_bound_df.copy()
            if score_col == "ARBS":
                plot_bound_df[score_col] = plot_bound_df[score_col].apply(log_modulus)
                plot_not_bound_df[score_col] = plot_not_bound_df[score_col].apply(log_modulus)

            # Prepare figure and axes
            fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300, sharex=True, sharey=True)
            cell_lines = self.cell_lines
            bound_statuses = [("Bound-Only", plot_bound_df), ("NOT-Bound-Only", plot_not_bound_df)]

            for row_idx, cell_line in enumerate(cell_lines):
                for col_idx, (status_label, df) in enumerate(bound_statuses):
                    ax = axes[row_idx, col_idx]
                    plot_df = df[df["Cell Line"] == cell_line].copy()
                    plot_df = plot_df.sort_values("Zero Cutoff")
                    # Use seaborn lineplot: x=Zero Cutoff, y=score_col, hue=Feature, style=Position
                    sns.lineplot(
                        data=plot_df,
                        x="Zero Cutoff",
                        y=score_col,
                        hue="Feature",
                        hue_order=plot_df["Feature"].unique(),
                        style="Position",
                        style_order=all_positions,
                        ax=ax,
                        legend=False,
                        markers=True,
                        dashes=True,
                        linewidth=0.5,
                        markersize=5,
                        alpha=0.5
                    )
                    ax.set_title(f"{cell_line} - {status_label}", fontsize=14)
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    
                    if score_col == 'NARBS': 
                        # Set y-axis limits for NARBS
                        ax.set_ylim(-1.05, 1.05)

                    ax.set_xscale("log")
                    ax.tick_params(axis="x", labelsize=14)
                    ax.tick_params(axis="y", labelsize=14)

            plt.suptitle(f"{score_col} vs Zero Cutoff per Feature\n", fontsize=18, y=1.01)
            fig.supxlabel("Zero Cutoff", fontsize=16, y=-0.01)
            fig.supylabel(score_col if score_col != "ARBS" else "log-modulus(ARBS)", fontsize=16, x=-0.01)
            plt.tight_layout()
            plt.show()


            # For each cutoff (excluding 0), plot score_col at cutoff=0 (x) vs score_col at cutoff!=0 (y)
            cutoffs = sorted(bound_df["Zero Cutoff"].unique())
            cutoffs_nonzero = [c for c in cutoffs if c != 0]
            for cutoff in cutoffs_nonzero:
                fig, axes = plt.subplots(2, 2, figsize=(8, 7), dpi=300, sharex=True, sharey=True)
                for row_idx, cell_line in enumerate(self.cell_lines):
                    for col_idx, (status_label, df) in enumerate([("Bound-Only", bound_df.copy()), ("NOT-Bound-Only", not_bound_df.copy())]):
                        ax = axes[row_idx, col_idx]
                        # Subset to this cell line and cutoff=0 and cutoff!=0
                        df0 = df[(df["Cell Line"] == cell_line) & (df["Zero Cutoff"] == 0)]
                        dfc = df[(df["Cell Line"] == cell_line) & (df["Zero Cutoff"] == cutoff)]
                        # Merge on Feature
                        merged = pd.merge(
                            df0[["Feature", score_col]],
                            dfc[["Feature", score_col]],
                            on="Feature",
                            suffixes=("_0", f"_{cutoff}")
                        )
                        
                        # Remove rows with nulls in either column
                        merged = merged.dropna(subset=[f"{score_col}_0", f"{score_col}_{cutoff}"])
                        x = merged[f"{score_col}_0"]
                        y = merged[f"{score_col}_{cutoff}"]

                        if score_col == "ARBS":
                            # Apply log-modulus transformation and set zeros to NaN
                            x = x.apply(log_modulus)
                            y = y.apply(log_modulus)

                        pearson_corr, _ = pearsonr(x, y)
                        spearman_corr, _ = spearmanr(x, y)
                        num_points = len(merged)

                        sns.scatterplot(x=x, y=y, ax=ax, color="deepskyblue", edgecolor="black", alpha=0.3, s=5)

                        ax.plot([x.min(), x.max()],
                                [x.min(), x.max()],
                                color="red", linestyle="--", linewidth=1, label="y=x")
                        ax.set_title(f"{cell_line} - {status_label}", fontsize=10)
                        ax.text(
                            0.5, 0.02,
                            f"Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}\nPoints: {num_points}",
                            transform=ax.transAxes,
                            fontsize=9,
                            verticalalignment='bottom',
                            horizontalalignment='left'
                        )

                        ax.set_xlabel("")
                        ax.set_ylabel("")

                plt.suptitle(f"{score_col}: Absolute 0 Cutoff vs {cutoff}", fontsize=16)
                fig.supylabel(f"Cutoff: {cutoff}", fontsize=14, x=-0.01)
                fig.supxlabel("Absolute 0 Cutoff", fontsize=14, y=-0.01)
                plt.tight_layout()
                plt.show()
    

    def plot_activator_repressor_behavior_score_vs_percent_positive_and_negative(self, underlying_data=None): 
        assert underlying_data in ["Unique-Binding"], "underlying_data must be 'Unique-Binding' for this function"

        # Load ARBS/NARBS scores for Bound-Only and NOT-Bound-Only from cache
        bound_df = self.calculate_activator_repressor_behavior_score(binding_mode="Bound-Only", underlying_data=underlying_data)
        not_bound_df = self.calculate_activator_repressor_behavior_score(binding_mode="NOT-Bound-Only", underlying_data=underlying_data)

        # Load percent positive/negative local SHAP for Bound-Only and NOT-Bound-Only
        bound_percent = self.calculate_percent_positive_and_negative_local_SHAP_per_feature(mode="Bound-Only", underlying_data=underlying_data)
        not_bound_percent = self.calculate_percent_positive_and_negative_local_SHAP_per_feature(mode="NOT-Bound-Only", underlying_data=underlying_data)

        # Find shared zero cutoff values between ARBS/NARBS and percent positive/negative local SHAP
        bound_zero_cutoffs = set(bound_df["Zero Cutoff"].unique())
        not_bound_zero_cutoffs = set(not_bound_df["Zero Cutoff"].unique())
        bound_percent_zero_cutoffs = set(bound_percent.keys())
        not_bound_percent_zero_cutoffs = set(not_bound_percent.keys())

        shared_cutoffs = sorted(
            bound_zero_cutoffs & not_bound_zero_cutoffs & bound_percent_zero_cutoffs & not_bound_percent_zero_cutoffs
        )
        # Remove cutoffs that are equal to or larger than 0.1
        shared_cutoffs = [c for c in shared_cutoffs if c < 0.1]

        for metric in ["ARBS", "NARBS"]:
            for cutoff in shared_cutoffs:
                fig, axes = plt.subplots(2, 2, figsize=(14, 13), dpi=50, sharex=True, sharey=True)

                for row_idx, cell_line in enumerate(self.cell_lines):
                    for col_idx, (mode, df, percent_dict) in enumerate([
                        ("Bound-Only", bound_df, bound_percent),
                        ("NOT-Bound-Only", not_bound_df, not_bound_percent)
                    ]):
                        # Subset ARBS/NARBS for this cell line and cutoff
                        metric_df = df[(df["Cell Line"] == cell_line) & (df["Zero Cutoff"] == cutoff)]
                        # Get percent positive and negative DataFrames for this cell line and cutoff
                        percent_pos_df = percent_dict[cutoff]["positive"][cell_line]
                        percent_neg_df = percent_dict[cutoff]["negative"][cell_line]
                        # Melt percent pos/neg to long form and merge
                        pos_long = percent_pos_df.reset_index().melt(id_vars="index", var_name="RBP", value_name="Percent_Positive")
                        neg_long = percent_neg_df.reset_index().melt(id_vars="index", var_name="RBP", value_name="Percent_Negative")
                        percent_long = pd.merge(pos_long, neg_long, on=["index", "RBP"])
                        percent_long["Feature"] = percent_long["RBP"].astype(str) + "_" + percent_long["index"].astype(str)
                        # Merge with metric_df on Feature
                        merged = pd.merge(
                            percent_long,
                            metric_df[["Feature", metric]],
                            on="Feature",
                            how="inner"
                        )
                        # Drop rows with nulls in either axis
                        merged = merged.dropna(subset=["Percent_Positive", "Percent_Negative", metric])
                        # Plot: x = Percent_Positive - Percent_Negative, y = NARBS
                        x = merged["Percent_Positive"] - merged["Percent_Negative"]
                        y = merged[metric]
                        ax = axes[row_idx, col_idx]
                        
                        sns.scatterplot(x=x, y=y, ax=ax, color="deepskyblue", edgecolor="black", alpha=0.5, s=10)

                        # Move axes to the center
                        ax.spines['left'].set_position('zero')
                        ax.spines['bottom'].set_position('zero')
                        ax.spines['right'].set_color('none')
                        ax.spines['top'].set_color('none')
                        ax.xaxis.set_ticks_position('bottom')
                        ax.yaxis.set_ticks_position('left')

                        # Remove default axis labels
                        ax.set_xlabel("")
                        ax.set_ylabel("")

                        # Correlations
                        pearson_corr, _ = pearsonr(x, y)
                        spearman_corr, _ = spearmanr(x, y)
                        num_points = len(merged)

                        ax.set_title(f"{cell_line} - {mode}", fontsize=16, pad=20, color="green")
                        ax.text(
                            0.9, 0.2,
                            f"Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}\nPoints: {num_points}",
                            transform=ax.transAxes,
                            fontsize=12,
                            verticalalignment='bottom',
                            horizontalalignment='right'
                        )

                fig.supxlabel("% Positive - % Negative", fontsize=24, y=-0.02)
                fig.supylabel(metric, fontsize=24, x=-0.02)
                plt.suptitle(f"{metric} vs (% Positive - % Negative) Local SHAP\n\nZero Cutoff = {cutoff}\nNOTE 1: {underlying_data} data", fontsize =18, y=1.01)
                plt.tight_layout()
                plt.show()

        
    
    def load_final_SHAP_data(self, underlying_data=None, as_lazyframe=False):
        assert underlying_data in ["All-Data", "Unique-Binding"], "underlying_data must be 'All-Data' or 'Unique-Binding'"
        assert as_lazyframe in [True, False], "as_lazyframe must be True or False"

        if underlying_data == "Unique-Binding":
            logger.warning(f"REMINDER: 'All-Data' is used to create cache and {underlying_data} must be created each time from 'All-Data' cache.")

        k562_output_file = self.CACHE_INFO["final_SHAP_cache"]["All-Data"]["K562"]
        hepg2_output_file = self.CACHE_INFO["final_SHAP_cache"]["All-Data"]["HepG2"]
        
        if os.path.exists(k562_output_file) and os.path.exists(hepg2_output_file):
            logger.info("FROM CACHE: Final SHAP data already exists for both cell lines.")

            final_shap_data = {
                "K562": pl.scan_ipc(k562_output_file),
                "HepG2": pl.scan_ipc(hepg2_output_file)
            }

            if underlying_data == "Unique-Binding":
                for key in final_shap_data:
                    schema = final_shap_data[key].collect_schema().names()
                    binding_cols = [col for col in schema if col.endswith("_binding")]
                    final_shap_data[key] = final_shap_data[key].unique(subset=binding_cols, maintain_order=True, keep="first")

            for key in final_shap_data:
                final_shap_data[key] = final_shap_data[key].sort('index')
            
            if as_lazyframe: 
                return final_shap_data
            
            elif not as_lazyframe:

                for key in final_shap_data:
                    final_shap_data[key] = final_shap_data[key].collect()

                if underlying_data == "Unique-Binding":
                    self.final_unique_binding_SHAP_data = final_shap_data
                elif underlying_data == "All-Data":
                    self.final_all_data_SHAP_data = final_shap_data

                logger.success(f"Loaded final SHAP data from cache for {underlying_data} mode.")

        else: 
            
            for cell_line in self.cell_lines: 
                output_file = self.CACHE_INFO["final_SHAP_cache"]["All-Data"][cell_line]

                # Step 1: Retrieve 5 SHAP tables
                shap_dfs = self.retrieve_5_SHAP_tables_per_cell_line(cell_line, binding_unique="All-Data")

                # Step 2: Get the first lazyframe and collect non-SHAP columns
                shap_lazyframes = self.get_SHAP_data_as_lazyframe(cell_line)
                schema = shap_lazyframes[0].collect_schema().names()
                non_shap_cols = [col for col in schema if not col.endswith("_shap")]
                base_df = shap_lazyframes[0].select(non_shap_cols).sort("index").collect()

                # Step 3: Remove 'has_RBP_KD' column
                base_df = base_df.drop("has_RBP_KD")

                # Step 4: Join with corrected has_RBP_KD results
                joined_df = self.get_has_RBP_KD_results(base_df, cell_line)
                assert joined_df.shape[0] == base_df.shape[0], "Row count changed after join"
                
                del base_df
                gc.collect()

                joined_df = joined_df.sort("index")

                # Step 5: Add "Binding Sum" column as the horizontal sum of all binding columns per row
                binding_cols = [col for col in joined_df.columns if col.endswith("_binding")]
                joined_df = joined_df.with_columns(
                    pl.sum_horizontal([pl.col(col) for col in binding_cols]).alias("Binding Sum")
                )

                # Step 6: Add "Averaged Prediction" columns
                joined_df = joined_df.drop("Predictions")

                indices_to_match = shap_dfs[0].sort("index")["index"].to_list()
                predictions_list = []
                for lf in shap_lazyframes:
                    pred_df = lf.select(["index", "Predictions"]).sort("index").collect()
                    assert pred_df["index"].to_list() == indices_to_match, "Indices do not match across SHAP dataframes"
                    predictions_list.append(pred_df["Predictions"].to_numpy())
                
                del pred_df
                gc.collect()
                
                # Add averaged predictions as new column
                joined_df = joined_df.with_columns(
                    pl.Series("Averaged Prediction (Probability)", np.mean(np.stack(predictions_list, axis=0), axis=0))
                )

                # Add averaged log-odds predictions as new column
                log_odds_predictions = [self.log_odds(pred) for pred in predictions_list]
                joined_df = joined_df.with_columns(
                    pl.Series("Averaged Prediction (Log-Odds)", np.mean(np.stack(log_odds_predictions, axis=0), axis=0))
                )
                
                del predictions_list, log_odds_predictions
                gc.collect()

                # Step 7: Add mean SHAP columns (order matches by sorted index)
                mean_df = self.calculate_pointwise_SHAP_metric_per_cell_line(cell_line_shap=shap_dfs, metric='mean')
                assert mean_df.shape[0] == joined_df.shape[0], "Row count mismatch after calculating mean SHAP"
                
                for col in mean_df.columns:
                    assert col.endswith("_shap"), f"Column {col} is not a SHAP column"
                    assert len(mean_df[col]) == len(joined_df), f"Length mismatch for column {col}"
                    joined_df = joined_df.with_columns(pl.Series(col, mean_df[col]))

                del mean_df, shap_dfs
                gc.collect()

                # Step 8: Assert that there are no missing or null values in the joined DataFrame
                assert joined_df.null_count().sum_horizontal().item() == 0, "Null values found in the final joined DataFrame"

                # Step 9: Write to LZ4-compressed feather file
                joined_df.sort('index').write_ipc(output_file, compression="lz4")
                logger.success(f"Saved final SHAP cache for {cell_line} to {output_file}")

                del joined_df
                gc.collect()


    def plot_local_SHAP_vs_PSI(self, feature=None, underlying_data=None):
        assert feature.endswith("_shap"), "feature must end with '_shap'"
        assert underlying_data in ["All-Data", "Unique-Binding"], "underlying_data must be 'All-Data' or 'Unique-Binding'"

        if not hasattr(self, "final_all_data_SHAP_data"): 
            self.load_final_SHAP_data(data_mode="All-Data", as_lazyframe=False)

        # Determine RBPs and position from feature
        rbp, position = self.get_RBP_position(feature)
        binding_col = feature.replace("_shap", "_binding")
        shap_col = feature
        psi_col = "Target_PSI"
        binding_sum_col = "Binding Sum"
        has_rbp_kd_col = f"has_RBP_KD_{position}"

        # Prepare memory-efficient DataFrames for plotting (vectorized)
        psi_dfs = []
        shap_dfs = []
        for cell_line in self.cell_lines:
            df = self.final_SHAP_data[cell_line]

            # Define masks and labels
            masks_labels = [
                (df[binding_col] == 1, "Bound"),
                (df[binding_col] == 0, "Not Bound"),
                ((df[binding_col] == 1) & (df[binding_sum_col] == 1), "Solo Binding"),
                ((df[binding_sum_col] == 0) & (df[has_rbp_kd_col] == True) & (df["RBP_KD_Target"] == rbp), "KD @ Pos"),
            ]

            for mask, label in masks_labels:
                filtered = df.filter(mask)

                if filtered.height == 0:
                    continue

                psi_df = pd.DataFrame({
                    "Cell Line": cell_line,
                    "Label": label,
                    "PSI": filtered[psi_col].to_numpy()
                })
                shap_df = pd.DataFrame({
                    "Cell Line": cell_line,
                    "Label": label,
                    "SHAP": filtered[shap_col].to_numpy()
                })
                psi_dfs.append(psi_df)
                shap_dfs.append(shap_df)

        psi_df = pd.concat(psi_dfs, ignore_index=True)
        shap_df = pd.concat(shap_dfs, ignore_index=True)

        # Plotting
        ncols = len(psi_df["Cell Line"].unique())
        nrows = 2

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 8), dpi=200, sharex=True, sharey='row')
        if ncols == 1:
            axes = np.array(axes).reshape(2, 1)

        for col_idx, cell_line in enumerate(sorted(psi_df["Cell Line"].unique())):
            # Top: Target_PSI
            ax_psi = axes[0, col_idx]
            plot_psi = psi_df[psi_df["Cell Line"] == cell_line]
            # Violinplot with light blue
            sns.violinplot(
                data=plot_psi, x="Label", y="PSI", ax=ax_psi, inner=None,
                color="#ADD8E6", cut=0, density_norm='width'
            )
            # Boxplot with small outlier points
            sns.boxplot(
                data=plot_psi, x="Label", y="PSI", ax=ax_psi, width=0.2, showcaps=True,
                showfliers=True, boxprops={"facecolor": "none"}, meanline=True, showmeans=True,
                meanprops={"color": "red"}, flierprops={"marker": "o", "markersize": 2, "markerfacecolor": "gray", "alpha": 0.5}
            )
            ax_psi.set_title(f"{cell_line} - PSI", fontsize=14)
            ax_psi.set_ylabel("Target_PSI", fontsize=12)
            ax_psi.set_xlabel("")
            ax_psi.set_xticklabels([])  # Remove individual x-axis labels
            # Set y-limit to 1.2 and annotate counts above each violin
            ax_psi.set_ylim(0, 1.2)
            for tick, label in enumerate(ax_psi.get_xticklabels()):
                cat = label.get_text()
                n_points = plot_psi[plot_psi["Label"] == cat].shape[0]
                ax_psi.text(
                    tick, 1.12, f"n={n_points}", ha="center", va="bottom",
                    fontsize=10, color="green"
                )

            # Bottom: Local SHAP
            ax_shap = axes[1, col_idx]
            plot_shap = shap_df[shap_df["Cell Line"] == cell_line]
            # Violinplot with light orange
            sns.violinplot(
                data=plot_shap, x="Label", y="SHAP", ax=ax_shap, inner=None,
                color="#FFD580", cut=0, density_norm='width'
            )
            # Boxplot with small outlier points
            sns.boxplot(
                data=plot_shap, x="Label", y="SHAP", ax=ax_shap, width=0.2, showcaps=True,
                showfliers=True, boxprops={"facecolor": "none"}, meanline=True, showmeans=True,
                meanprops={"color": "red"}, flierprops={"marker": "o", "markersize": 2, "markerfacecolor": "gray", "alpha": 0.5}
            )
            ax_shap.set_title(f"{cell_line} - Local SHAP", fontsize=14)
            ax_shap.set_ylabel("Local SHAP", fontsize=12)
            ax_shap.set_xlabel("")
            ax_shap.set_xticklabels([])  # Remove individual x-axis labels
            # Add gold line at y=0
            ax_shap.axhline(0, color="gold", linestyle="--", linewidth=1.5, zorder=0)

        fig.suptitle(
            f"Local SHAP and PSI Distributions for {feature}\nNOTE 1: {underlying_data} shown here.",
            fontsize=16
        )
        fig.supxlabel("Category", fontsize=13)
        plt.tight_layout()
        plt.show()


    def plot_actual_vs_predicted_for_best_models(self, underlying_data = None): 

        assert underlying_data in ["All-Data", "Unique-Binding"], "underlying_data must be 'All-Data' or 'Unique-Binding'"

        # Prepare data for all cell lines
        dfs = {}
        r2_scores = {}
        for cell_line, model_hash in self.XGBOOST_BEST_MODEL_HASHES.items():
            pred_file = os.path.join(self.PRED_DIR, "XGBRegressor", f"{model_hash}.feather")
            assert os.path.exists(pred_file), f"Prediction file not found: {pred_file}"

            # Load all columns needed for partitioning and plotting
            preds = pl.scan_ipc(pred_file).filter(pl.col("Partition") == "Test")

            if underlying_data == "Unique-Binding":
                # Get all columns ending with "_binding"
                binding_cols = [col for col in preds.collect_schema().names() if col.endswith("_binding")]
                # Group by all binding columns and aggregate mean of Target_PSI and Predictions
                preds = preds.group_by(binding_cols).agg(
                    [
                        pl.col("Target_PSI").mean().alias("Target_PSI"),
                        pl.col("Predictions").mean().alias("Predictions"),
                    ]
                )

            preds = preds.select(["Predictions", "Target_PSI"]).collect()

            y_true = preds["Target_PSI"].to_numpy()
            y_pred = preds["Predictions"].to_numpy()
            dfs[cell_line] = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

            # Calculate R2 score only for the "Test" partition
            r2_scores[cell_line] = r2_score(y_true, y_pred)

            del preds
            gc.collect()

        # Set up a single figure with subplots for each cell line
        n = len(dfs)
        fig = plt.figure(figsize=(7 * n, 7), dpi=100)
        gs = gridspec.GridSpec(2, n, height_ratios=[1, 4], hspace=0.25, wspace=0.25)

        hexbin_objs = []
        for idx, (cell_line, df) in enumerate(dfs.items()):
            # Main scatter/hexbin plot
            ax_joint = fig.add_subplot(gs[1, idx])
            hb = ax_joint.hexbin(
                df["y_true"], df["y_pred"],
                gridsize=100, cmap="Blues", norm=LogNorm(), mincnt=1
            )
            hexbin_objs.append(hb)

            # Annotate R2 score and number of points at the center top (Test partition only)
            ax_joint.text(
                0.5, 0.97,
                f"$R^2$: {r2_scores[cell_line]:.3f}\nPoints: {len(df):.2e}",
                transform=ax_joint.transAxes,
                fontsize=14, color="red",
                ha="center", va="top"
            )

            # Add line from (0,1) to (0,1)
            ax_joint.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1, label="y=x")

            # ax_joint.set_ylabel("Predicted PSI", fontsize=12)
            ax_joint.set_title(f"{cell_line}", fontsize=18)
            # ax_joint.legend(fontsize=10, loc="upper left")

            # Marginal histogram for x (top)
            ax_histx = fig.add_subplot(gs[0, idx], sharex=ax_joint)
            ax_histx.hist(df["y_true"], bins=50, color="#4682B4", alpha=0.7, density=True, edgecolor="black")
            sns.kdeplot(df["y_true"], color="orange", lw=1, ax=ax_histx)
            ax_histx.set_xlim(0, 1)
            ax_histx.axis("off")
            # Move the axis slightly down
            pos = ax_histx.get_position()
            ax_histx.set_position([pos.x0, pos.y0 - 0.03, pos.width, pos.height])

            # Marginal histogram for y (right)
            ax_histy = ax_joint.inset_axes([1.02, 0, 0.15, 1], sharey=ax_joint)
            ax_histy.hist(df["y_pred"], bins=50, color="#4682B4", alpha=0.7, orientation="horizontal", density=True, edgecolor="black")
            sns.kdeplot(df["y_pred"], color="orange", lw=1, ax=ax_histy, vertical=True)
            ax_histy.set_ylim(0, 1)
            ax_histy.axis("off")

        # Add a separate horizontal colorbar below each joint subplot
        for idx in range(n):
            ax_joint = fig.axes[2 * idx + 1]  # axes are [histx0, joint0, histx1, joint1, ...]
            # Get the position of the joint axes in figure coordinates
            pos = ax_joint.get_position()
            # Manually specify the colorbar axes below the joint plot
            cbar_height = 0.03
            cbar_pad = 0.65
            cbar_ax = fig.add_axes([
                pos.x0,
                pos.y0 - cbar_height - cbar_pad,  # ensure it's below the subplot
                pos.width,
                cbar_height
            ])

            fig.colorbar(
                hexbin_objs[idx],  # Use the PolyCollection from hexbin
                cax=cbar_ax,
                orientation='horizontal'
            )
            cbar_ax.set_xlabel('Counts (log scale)', fontsize=12)

        fig.suptitle(f"Test Partition: Actual vs Predicted PSI\nNOTE: showing top model per cell line based on outer holdout $R^2$\nNOTE 2: data mode is {underlying_data}", fontsize=22, y=1.03)
        fig.supxlabel("Actual PSI", fontsize=20, y=-0.07)
        fig.supylabel("Predicted PSI", fontsize=20, x=0.06, y=0.4)
        plt.tight_layout()

        plt.savefig(self.FIGURES["predicted_vs_actual_PSI_plot"][underlying_data], dpi=600, bbox_inches='tight')
        plt.show()

        del dfs 
        gc.collect()


    def plot_global_SHAP_distributions_across_binding_modes(self): 
        # Load global SHAP values for 5_dfs_average, Bound-Only, and NOT-Bound-Only (all with Unique-Binding)
        global_shap_5dfs = self.calculate_global_SHAP(mode="5_dfs_average", binding_unique="Unique-Binding")
        global_shap_bound = self.calculate_specialized_global_SHAP(mode="Bound-Only", condition=None, underlying_data="Unique-Binding")
        global_shap_unbound = self.calculate_specialized_global_SHAP(mode="NOT-Bound-Only", condition=None, underlying_data="Unique-Binding")

        # Combine all into a long table: columns = cell line, feature, global shap, type
        long_data = []
        for shap_type, shap_dict in [
            ("5_dfs_average", global_shap_5dfs),
            ("unbound", global_shap_unbound),
            ("bound", global_shap_bound),
        ]:
            for cell_line, heatmap in shap_dict.items():
                for pos in heatmap.index:
                    for rbp in heatmap.columns:
                        feature = f"{rbp}_{pos}"
                        value = heatmap.at[pos, rbp]
                        long_data.append({
                            "Cell Line": cell_line,
                            "Feature": feature,
                            "Global SHAP": value,
                            "Type": shap_type,
                        })
        df_long = pd.DataFrame(long_data)
        # Drop rows with null values in "Global SHAP"
        df_long = df_long.dropna(subset=["Global SHAP"])

        # Set categorical order for Type and Cell Line
        type_order = ["5_dfs_average", "unbound", "bound"]
        type_labels = {"5_dfs_average": "All", "unbound": "Unbound", "bound": "Bound"}
        df_long["Type"] = pd.Categorical(df_long["Type"], categories=type_order, ordered=True)
        df_long["Binding Mode"] = df_long["Type"].map(type_labels)
        cell_line_order = ["HepG2", "K562"]
        df_long["Cell Line"] = pd.Categorical(df_long["Cell Line"], categories=cell_line_order, ordered=True)

        # Prepare colors
        palette = {"HepG2": "#FFD580", "K562": "#ADD8E6"}

        plt.figure(figsize=(9, 5), dpi=300)
        ax = plt.gca()

        # Violinplot with boxplot inside, grouped by Type and split by Cell Line
        sns.violinplot(
            data=df_long,
            x="Binding Mode",
            y="Global SHAP",
            hue="Cell Line",
            order=[type_labels[t] for t in type_order],
            hue_order=cell_line_order,
            palette=palette,
            cut=0,
            linewidth=1,
            density_norm="width",
            ax=ax,
            split=False,
            inner="point",
            inner_kws={"marker": "o", "alpha": 0.9, "color": "lightgreen",},
        )

        annot = Annotator(
            ax, 
            pairs = [
                (("All", "HepG2"), ("Bound", "HepG2")),
                (("All", "K562"), ("Bound", "K562")),
                (("Unbound", "HepG2"), ("Bound", "HepG2")),
                (("Unbound", "K562"), ("Bound", "K562")),
            ], 
            data = df_long,
            x = "Binding Mode",
            y = "Global SHAP",
            hue = "Cell Line",
            order = [type_labels[t] for t in type_order],
            hue_order = cell_line_order,
        )
        annot.configure(
            test="Mann-Whitney-ls", 
            text_format="star", 
            color='salmon',
            loc="inside", 
            verbose=2, 
            comparisons_correction='fdr_bh',
            text_offset = 1, 
            line_height = 0.05
        )
        annot.apply_test().annotate(
            line_offset_to_group = 0.3
        )
        
        # Remove duplicate legends
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], title="Cell Line", loc="center left", fontsize=10, title_fontsize=12)

        # Annotate number of points, median, and percent zero above each violin
        for i, type_label in enumerate([type_labels[t] for t in type_order]):
            for j, cell_line in enumerate(cell_line_order):
                vals = df_long[(df_long["Binding Mode"] == type_label) & (df_long["Cell Line"] == cell_line)]["Global SHAP"].dropna()
                n_points = len(vals)
                median_val = np.median(vals)
                pct_zero = (vals == 0).mean() * 100
                x_pos = i - 0.2 + j * 0.4  # violinplot offset
                y_pos = vals.max() + 0.02
                ax.text(
                    x_pos, y_pos,
                    f"#: {n_points}\nMedian: {median_val:.1e}\n% Zero: {pct_zero:.0f}",
                    ha="center", va="bottom", fontsize=8, color="black"
                )
        
        ax.set_title(f"{self.latex_symbols['Unique-Binding']['5_dfs_average']} Distribution by Binding Mode and Cell Line\nNOTE 1: Using 'Unique Binding'\nNOTE 2: Mann-Whitney U test checks 'Bound' is greater than other distribution", fontsize=8, y=1.05)
        ax.set_xlabel("Binding Mode", fontsize=14, y=-0.05)
        ax.set_ylabel(self.latex_symbols["Unique-Binding"]["5_dfs_average"], fontsize=20)

        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=14)

        # Remove top and right spines for a cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.FIGURES["global_SHAP_distribution"]["All Together"], dpi=600, bbox_inches='tight')
        plt.show()


    def plot_global_SHAP_examples_as_bar_plots(self, mode=None, underlying_data=None):
        assert underlying_data in ["All-Data", "Unique-Binding"], "underlying_data must be 'All-Data' or 'Unique-Binding'"
        assert mode in ["Bound-Only", "Signed-Local-SHAP-Mean-Bound-Only"], "mode must be 'Bound-Only' or 'Signed-Local-SHAP-Mean-Bound-Only'"

        hand_selected_rbps = {
            "TARDBP": [1],
            "RBFOX2": [4],
            "FTO": [5],
            "DDX55": [2, 4],
            "RPS3": [3, 4, 6],
            "GRWD1": [3, 4, 6],
            "DDX3X": [3, 4, 5, 6],

            "HNRNPM": [3, 4],
            "IGF2BP1": [3, 4],
            "LIN28B": [3, 4],
            "SND1": [3, 4],
            "FXR2": [3, 4],

        }


        global_shap = self.calculate_specialized_global_SHAP(
            mode=mode, condition=None, underlying_data=underlying_data
        )

        # Separate RBPs into two groups: those with [3, 4] and those with other highlighted positions
        rbps_3_4 = [rbp for rbp, pos_list in hand_selected_rbps.items() if pos_list == [3, 4]]
        rbps_other = [rbp for rbp, pos_list in hand_selected_rbps.items() if pos_list != [3, 4]]
    
        if mode == "Bound-Only":
            # for rbp in hand_selected_rbps:
            #     fig, axes = plt.subplots(2, 1, figsize=(4,4.5), dpi=150, sharex=True)
            #     for idx, cell_line in enumerate(self.cell_lines):
            #         heatmap = global_shap[cell_line]
            #         ax = axes[idx]
            #         y = [heatmap.at[pos, rbp] for pos in range(1, 7)]
            #         x = np.arange(1, 7)
            #         # Plot the main stem plot for all positions
            #         markerline, stemlines, baseline = ax.stem(
            #             x, y, basefmt=" ", markerfmt="o",
            #         )
            #         plt.setp(markerline, markersize=5, color="#7570b3")
            #         plt.setp(stemlines, color="#7570b3", linestyle=":")  # Make the stem lines dotted

            #         # Plot a second stem plot for highlighted positions using color #1b9e77
            #         highlight_positions = hand_selected_rbps[rbp]
            #         highlight_x = np.array(highlight_positions)
            #         highlight_y = [heatmap.at[pos, rbp] for pos in highlight_x]
            #         markerline2, stemlines2, baseline2 = ax.stem(
            #             highlight_x, highlight_y, basefmt=" ", markerfmt="o",
            #         )
            #         plt.setp(markerline2, markersize=7, color="#00ffb3")
            #         plt.setp(stemlines2, color="#00ffb3", linestyle=":")  # Make the stem lines dotted

            #         for xi, yi in zip(x, y):
            #             ax.text(
            #                 xi, yi + 0.08 * np.nanmax(y),
            #                 f"{yi:.2f}", ha="center", va="bottom", fontsize=8, color ="#fc6f03"

            #             )

            #         ax.set_title(
            #             cell_line,
            #             fontsize=10
            #         )
            #         ymin, ymax = ax.get_ylim()
            #         ax.set_ylim(0- (0.1*(ymax-ymin)), ymax + 0.25 * (ymax - ymin))

            #     fig.supxlabel("Position", fontsize=12, x=0.6, y=0.05)
            #     fig.supylabel(self.latex_symbols[underlying_data][mode], fontsize=14, x=0.07)
            #     fig.suptitle(
            #         f"{rbp}: {self.latex_symbols[underlying_data][mode]} Across Positions\nNOTE 1: {underlying_data} data\nNOTE 2: hand-selected positions highlighted in green",
            #         fontsize=8, y=0.97
            #     )

            #     plt.tight_layout()
            #     plt.show()
        

            # Prepare plotting data for both groups
            rbp_groups = [
                {"rbps": rbps_other, "type": "pos_other_highlight", 'supylabel_y_position': -0.08, 'sharey': True},
                {"rbps": rbps_3_4, "type": "pos_3_4_highlight", 'supylabel_y_position': 0.015, 'sharey': True},
            ]

            for group in rbp_groups:
                rbps = group["rbps"]
        
                nrows = len(rbps)
                ncols = 2  # One for each cell line

                fig, axes = plt.subplots(
                    nrows=nrows, ncols=ncols, figsize=(1.8 * ncols, 1 * nrows),
                    sharex=True, sharey=group['sharey'], squeeze=False, gridspec_kw={'hspace': 0.05, 'wspace': 0.3}, dpi=300
                )

                for row_idx, rbp in enumerate(rbps):
                    highlight_positions = hand_selected_rbps[rbp]
                    for col_idx, cell_line in enumerate(self.cell_lines):
                        ax = axes[row_idx, col_idx]
                        heatmap = global_shap[cell_line]
                        y = [heatmap.at[pos, rbp] for pos in range(1, 7)]
                        x = np.arange(1, 7)
                        colors = ["#00ffb3" if (pos in highlight_positions) else "#7570b3" for pos in x]

                        ax.bar(x, y, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
                        # Make y-axis tick labels smaller
                        ax.tick_params(axis='y', labelsize=10, color='red')
                    
                        ax.set_yticklabels([f"{tick:.1f}" for tick in ax.get_yticks()], fontsize=8, )

                        if col_idx == 0:
                            ax.set_ylabel(f"$\\bf{{{rbp}}}$", fontsize=12, rotation=0, labelpad=10, va='center', y=0.3, ha='right')
                        else:
                            ax.set_ylabel("")
                        ax.set_xlabel("")

                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_visible(False)

                # Set shared x-axis label only on the bottom row
                for col_idx in range(ncols):
                    axes[-1, col_idx].set_xticks(np.arange(1, 7))
                    axes[-1, col_idx].set_xticklabels([str(i) for i in range(1, 7)], fontsize=7)

                fig.supxlabel("Position", fontsize=14, y=group['supylabel_y_position'])
                fig.supylabel(self.latex_symbols[underlying_data][mode], fontsize=20, x=-0.32)
                for ax in axes.flat:
                    ax.tick_params(axis='x', labelsize=10)

                # Add cell line labels as column titles
                for col_idx, cell_line in enumerate(self.cell_lines):
                    axes[0, col_idx].set_title(cell_line, fontsize=13, pad=10)

                plt.tight_layout(pad=2)

                plt.savefig(self.FIGURES["global_shap_position_highlighting_bar_plots"][mode][group["type"]], dpi=600, bbox_inches='tight')
                plt.show()
            
        elif mode == "Signed-Local-SHAP-Mean-Bound-Only":

            # Prepare plotting data for both groups
            # Move "HNRNPM" to the end of rbps_3_4 as it is the only negative one
            rbps_3_4_ordered = [rbp for rbp in rbps_3_4 if rbp != "HNRNPM"] + ["HNRNPM"]
            rbp_groups = [
                {"rbps": rbps_other, "type": "pos_other_highlight", 'sharey': False},
                {"rbps": rbps_3_4_ordered, "type": "pos_3_4_highlight", 'sharey': False},
            ]

            for group in rbp_groups:
                rbps = group["rbps"]
                nrows = len(rbps)
                ncols = 1  # Only one column, both cell lines per subplot

                # Gather all values to determine global min/max for y-axis
                all_vals = []
                for rbp in rbps:
                    for pos in range(1, 7):
                        for cell_line in self.cell_lines:
                            val = global_shap[cell_line].at[pos, rbp]
                            if not pd.isna(val):
                                all_vals.append(val)

                fig, axes = plt.subplots(
                    nrows=nrows, ncols=ncols, figsize=(5.5, 1.5 * nrows),  # (1) increase x width
                    sharex=True, sharey=group["sharey"], squeeze=False,
                    gridspec_kw={'hspace': 0.2,}, dpi=300
                )

                for row_idx, rbp in enumerate(rbps):
                    ax = axes[row_idx, 0]
                    highlight_positions = hand_selected_rbps[rbp]
                    x = np.arange(1, 7)
                    bar_width = 0.3
                    gap = 0.15

                    # Gather all bar values for this subplot to determine y-range
                    all_bar_vals = []
                    for pos in x:
                        for ci, cell_line in enumerate(self.cell_lines):
                            val = global_shap[cell_line].at[pos, rbp]
                            if not pd.isna(val):
                                all_bar_vals.append(val)
                    
                    y_min = min(all_bar_vals)
                    y_max = max(all_bar_vals)
                    y_range = y_max - y_min
                    y_expand = y_range * 0.25
                    y_offset = y_range * 0.05
                    ax.set_ylim(y_min - y_expand, y_max + y_expand)

                    # Plot bars for both cell lines at each position
                    for pos in x:
                        for ci, cell_line in enumerate(self.cell_lines):
                            val = global_shap[cell_line].at[pos, rbp]
                            if pd.isna(val):
                                continue
                            offset = -bar_width/2 - gap/2 if ci == 0 else bar_width/2 + gap/2
                            hatch = None if ci == 0 else 'x**'  # (3) change hatch pattern
                            facecolor = "white"
                            edgecolor = "black"
                            linewidth = 1.5
                            # Highlighted positions: fill with color if not white
                            if pos in highlight_positions:
                                if val > 0:
                                    facecolor = "#c83a3cff"
                                    txt_color = "#c83a3cff"
                                elif val < 0:
                                    facecolor = "#3a85bfff"
                                    txt_color = "#3a85bfff"
                                else:
                                    txt_color = "black"
                            else:
                                txt_color = "black"
                            # For hatches, do not set a color (2)
                            if hatch:
                                ax.bar(
                                    pos + offset, val, width=bar_width,
                                    color=facecolor, edgecolor=edgecolor, linewidth=linewidth,
                                    hatch=hatch, zorder=2
                                )
                            else:
                                ax.bar(
                                    pos + offset, val, width=bar_width,
                                    color=facecolor, edgecolor=edgecolor, linewidth=linewidth,
                                    zorder=2
                                )
                            # (8) Only add text for highlighted bars
                            if not pd.isna(val) and pos in highlight_positions:
                                if val > 0:
                                    y_txt = val + y_offset
                                    va = "bottom"
                                else:
                                    y_txt = val - y_offset
                                    va = "top"
                                ax.text(
                                    pos + offset, y_txt, f"{val:.2f}",
                                    ha="center", va=va,
                                    fontsize=9, color=txt_color
                                )

                    # Add a horizontal dotted line at y=0 with yellow-like color 
                    ax.axhline(0, color="#bbbb62", linestyle=":", linewidth=2, zorder=1)

                    # Row label: bold, right-aligned, centered vertically
                    ax.set_ylabel(rbp, fontsize=20, fontweight="bold", rotation=0, labelpad=12, va='center', ha='right', y=0.5)
                    ax.yaxis.set_label_coords(-0.03, 0.5)
                    # Remove y-axis ticks and labels except for the first subplot
                    ax.set_yticks([])
                    # Remove spines except left and bottom
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    # Only show x-axis ticks on the bottom subplot
                    if row_idx != nrows - 1:
                        ax.set_xticklabels([])
                    else:
                        ax.set_xticks(x, )
                        ax.set_xticklabels([str(i) for i in x], fontsize=25)
                        ax.set_xlabel("")

                # Add shared y-axis label and adjust layout (moved outside the rbp loop)
                fig.supylabel(self.latex_symbols[underlying_data][mode], fontsize=40, x=-0.3)
                fig.supxlabel("Position", fontsize=30, y=-0.03 if group['type'] == "pos_3_4_highlight" else 0)

                # Add legend for cell lines with hatch patterns (moved outside the rbp loop)
                legend_handles = []
                for ci, cell_line in enumerate(self.cell_lines):
                    hatch = None if ci == 0 else 'x**'
                    patch = Patch(
                        facecolor='white',
                        edgecolor='black',
                        hatch=hatch,
                        label=cell_line,
                        linewidth=1.8
                    )
                    legend_handles.append(patch)

                legend = fig.legend(
                    handles=legend_handles,
                    labels=self.cell_lines,
                    loc="center right",
                    bbox_to_anchor=(1.32, 0.5),
                    fontsize=18,
                    frameon=True,
                    borderaxespad=1.0,
                )

                legend.get_frame().set_linewidth(1.5)
                legend.get_frame().set_edgecolor('black')

                plt.tight_layout()
                plt.savefig(
                    self.FIGURES["global_shap_position_highlighting_bar_plots"][mode][group["type"]],
                    dpi=600, bbox_inches='tight'
                )
                plt.show()


    def plot_binding_sum_distribution(self):
        lf = self.load_final_SHAP_data(underlying_data="Unique-Binding", as_lazyframe=True)

        # Prepare a DataFrame for violinplot (percentage)
        plot_data = []
        for cell_line in self.cell_lines:
            binding_sum = lf[cell_line].select("Binding Sum").collect().to_pandas()["Binding Sum"].to_numpy()
            
            # Count number of columns ending with "_binding"
            num_binding_cols = len([col for col in lf[cell_line].collect_schema().names() if col.endswith("_binding")])

            binding_sum_pct = (binding_sum / num_binding_cols) * 100
            plot_data.append(pd.DataFrame({
                "Cell Line": cell_line,
                "Binding Sum (%)": binding_sum_pct,
                "Binding Sum": binding_sum
            }))

        plot_df = pd.concat(plot_data, ignore_index=True)

        for col in ["Binding Sum", "Binding Sum (%)"]:
            plt.figure(figsize=(8, 5), dpi=150)
            sns.violinplot(
                data=plot_df,
                x="Cell Line",
                y=col,
                inner="box",
                cut=0,
                scale="width",
                palette="pastel"
            )
            plt.xlabel("Cell Line", fontsize=14)
            plt.ylabel(col, fontsize=14)
            plt.title(f"{col} Across Cell Lines (Unique-Binding)", fontsize=16)
            plt.tight_layout()
            plt.show()


    def calculate_num_and_percent_bound_local_shap_greater_than_cutoff(self): 

        if os.path.exists(self.CACHE_INFO["per_row_num_and_percent_greater_than_cutoff"]["HepG2"]) and os.path.exists(self.CACHE_INFO["per_row_num_and_percent_greater_than_cutoff"]["K562"]):
            logger.success("FROM CACHE: loading...")

            return {
                    "HepG2": pl.read_csv(self.CACHE_INFO["per_row_num_and_percent_greater_than_cutoff"]["HepG2"], separator="\t"),
                    "K562": pl.read_csv(self.CACHE_INFO["per_row_num_and_percent_greater_than_cutoff"]["K562"], separator="\t")
            }
            
        else: 
            logger.info("No cache found. Calculating...")
            lazyframe = self.load_final_SHAP_data(underlying_data="Unique-Binding", as_lazyframe=True)

            data = {}
            for key in lazyframe.keys():
                schema = lazyframe[key].collect_schema().names()
                binding_cols = [col for col in schema if col.endswith("_binding")]

                select_cols = ["index", "Binding Sum"] + binding_cols + [f"{col.replace('_binding', '_shap')}" for col in binding_cols]
                data[key] = lazyframe[key].select(select_cols).collect()

            cutoffs = [0.01, .1, .5, 1]

            for cell_line, df in data.items():
                # Prepare binding and shap columns
                binding_cols = [col for col in df.columns if col.endswith("_binding")]
                shap_cols = [col.replace("_binding", "_shap") for col in binding_cols]

                # Prepare output columns
                output_cols = [
                    "index", "feature_num_bound", "rbp_num_bound"
                ]
                for cutoff in cutoffs:
                    output_cols.extend([
                        f"feature_num_shap_gt_{cutoff}",
                        f"feature_pct_shap_gt_{cutoff}",
                        f"rbp_num_shap_gt_{cutoff}",
                        f"rbp_pct_shap_gt_{cutoff}"
                    ])

                # Prepare lists to collect results
                results = []

                # Convert to numpy for speed
                binding_arr = df.select(binding_cols).to_numpy()
                shap_arr = np.abs(df.select(shap_cols).to_numpy())
                indices = df["index"].to_numpy()
                binding_sum_arr = df["Binding Sum"].to_numpy()

                # Precompute RBP and position for each feature column
                feature_rbps = [self.get_RBP_position(col)[0] for col in binding_cols]

                for i in range(binding_arr.shape[0]):
                    bound_mask = binding_arr[i] == 1
                    num_bound = bound_mask.sum()
                    # Assert num_bound matches Binding Sum
                    assert num_bound == binding_sum_arr[i], f"Row {indices[i]}: num_bound {num_bound} != Binding Sum {binding_sum_arr[i]}"
                    
                    # Get RBPs for bound features
                    rbp_num_bound = len(
                        set(
                            [feature_rbps[j] for j, is_bound in enumerate(bound_mask) if is_bound]
                        )
                    )

                    row_result = [indices[i], num_bound, rbp_num_bound]
                    # For each cutoff, calculate feature-based and RBP-based stats
                    for cutoff in cutoffs:
                        if num_bound == 0:
                            row_result.extend([float('nan'), float('nan'), float('nan'), float('nan')])
                        else:
                            # Find indices of bound features
                            bound_indices = np.where(bound_mask)[0]

                            # Find indices among bound features where SHAP > cutoff
                            gt_indices = [j for j in bound_indices if shap_arr[i][j] > cutoff]
                            num_gt = float(len(gt_indices))
                            pct_gt = float((num_gt / num_bound) * 100)
                            row_result.extend([num_gt, pct_gt])

                            # RBP-based: count unique RBPs with at least one feature's SHAP > cutoff
                            if rbp_num_bound == 0:
                                row_result.extend([float('nan'), float('nan')])
                            else:
                                rbps_with_shap_gt = set(feature_rbps[j] for j in gt_indices)
                                rbp_num_gt = float(len(rbps_with_shap_gt))
                                rbp_pct_gt = float((rbp_num_gt / rbp_num_bound) * 100)
                                row_result.extend([rbp_num_gt, rbp_pct_gt])

                    results.append(row_result)

                # Create polars DataFrame and save
                out_df = pl.DataFrame(results, schema=output_cols, orient="row")
                # Convert to pandas and write as gzipped TSV
                out_df.to_pandas().to_csv(self.CACHE_INFO["per_row_num_and_percent_greater_than_cutoff"][cell_line], sep="\t", index=False, compression="gzip")

                logger.success(f"Saved {cell_line} per-row num and percent greater than cutoff to {self.CACHE_INFO['per_row_num_and_percent_greater_than_cutoff'][cell_line]}")


    def plot_num_and_percent_bound_local_shap_greater_than_cutoff(self): 
        # # Load the cached data
        data = self.calculate_num_and_percent_bound_local_shap_greater_than_cutoff()

        cutoff_order = sorted(
            {
                float(col.split("_")[-1]) for col in data["K562"].columns if col.startswith("feature_num_shap_gt_")
            }
        )

        # Prepare data for violin plot
        plot_rows = []
        for cell_line, df in data.items():
            for col in df.columns:
                if "num_shap_gt_" in col or "pct_shap_gt_" in col: 
                    # Extract cutoff value from column name
                    cutoff = float(col.split("_")[-1])
                    # Determine type: "#" or "%"
                    value_type = "#" if "_num_" in col else "%"
                    unit = "Features" if col.startswith("feature_") else "RBPs"

                    # For each row, add to plot_rows
                    for val in df[col].to_numpy():
                        plot_rows.append({
                            "Cell Line": cell_line,
                            "Type": value_type,
                            "Cutoff": cutoff,
                            "Unit": unit,
                            "Value": val
                        })
        plot_df = pd.DataFrame(plot_rows)

        for cutoff in cutoff_order:
            for (x_axis, hue, x_order, hue_order) in [
                ("Unit", "Cell Line", ["Features", "RBPs"], self.cell_lines),
                ("Cell Line", "Unit", self.cell_lines, ["Features", "RBPs"])
            ]:
                fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300, sharey=False, sharex=True)
                for col_idx, value_type in enumerate(["#", "%"]):
                    ax = axes[col_idx]
                    plot_subset = plot_df[
                        (plot_df["Cutoff"] == cutoff) &
                        (plot_df["Type"] == value_type)
                    ]
                    sns.violinplot(
                        data=plot_subset,
                        x=x_axis,
                        y="Value",
                        hue=hue,
                        order=x_order,
                        hue_order=hue_order,
                        cut=0,
                        scale="width",
                        inner="box",
                        ax=ax
                    )

                    ax.set_title(f"{value_type}", fontsize=18)
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    
                    ax.tick_params(axis='x', labelsize=11)
                    ax.tick_params(axis='y', labelsize=11)

                    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
                    ax.minorticks_on()
                    ax.grid(True, which='minor', axis='y', linestyle=':', alpha=0.3)
                        
                plt.suptitle(
                    f"{self.latex_symbols['Unique-Binding']['local_SHAP']} > {cutoff}", 
                    fontsize=16, y=0.92
                )
                fig.supxlabel(x_axis, fontsize=16, x=0.5, y=0.02)
                fig.supylabel(f"# or % > {cutoff}", fontsize=16, y=0.5)
                # Add a single legend to the right of the figure
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(
                    handles, labels, title=hue,
                    bbox_to_anchor=(.98, 0.5), loc="center left", fontsize=12, title_fontsize=13
                )

                plt.tight_layout()
                if cutoff == .1 and x_axis == "Cell Line" and hue == "Unit":
                    plt.savefig(self.FIGURES["per_row_num_bound_and_percent_greater_than_cutoff"]["num_and_percentage_bound_local_SHAP_greater_than_0.1"], dpi=600, bbox_inches='tight')
                plt.show()


        nrows = len(self.cell_lines)
        ncols = len(cutoff_order)

        for col_type in ['feature', 'rbp']:
            for log_norm in [False, True]:
                fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows), dpi=300, sharex="row", sharey="row")
                for row_idx, cell_line in enumerate(self.cell_lines):
                    df = data[cell_line].to_pandas()
                    num_bound_col = f"{col_type}_num_bound"

                    max_num_bound = int(df[num_bound_col].max())
                    min_num_bound = int(df[num_bound_col].min())
                    # Ensure bins start at 1 or the minimum value, and step by 2
                    bin_start = max(1, min_num_bound)
                    bin_edges = np.arange(bin_start, max_num_bound + 2, 2)  # +2 to include the last bin

                    # First, determine the maximum count (vmax) for this row across all cutoffs
                    hist2d_counts = []
                    for col_idx, cutoff in enumerate(cutoff_order):
                        if cutoff == 1.0:
                            cutoff = 1
                        x = df[num_bound_col]
                        y = df[f"{col_type}_num_shap_gt_{cutoff}"]
                        # Compute the histogram counts only, using the custom bin_edges
                        counts, _, _ = np.histogram2d(
                            x, y, bins=[bin_edges, bin_edges]
                        )
                        hist2d_counts.append(counts)
                    row_vmax = max(np.max(c) for c in hist2d_counts if c.size > 0)

                    for col_idx, cutoff in enumerate(cutoff_order):
                        if cutoff == 1.0:
                            cutoff = 1

                        ax = axes[row_idx, col_idx]

                        x = df[num_bound_col]
                        y = df[f"{col_type}_num_shap_gt_{cutoff}"]

                        # Hexbin plot with logarithmic color scale, shared vmax within row, and thin black borders
                        h = ax.hexbin(
                            x, y, gridsize=[len(bin_edges)-1, len(bin_edges)-1],
                            cmap="Oranges",
                            extent=[bin_edges[0], bin_edges[-1], bin_edges[0], bin_edges[-1]],
                            norm=LogNorm(vmin=1, vmax=row_vmax) if log_norm else None,
                            edgecolors='black',
                            linewidths=0.05,
                            mincnt=1
                        )

                        # # Line of best fit
                        # mask = (~np.isnan(x)) & (~np.isnan(y))
                        # slope, intercept = np.polyfit(x[mask], y[mask], 1)

                        # x_fit = np.linspace(bin_edges[0], bin_edges[-1], 100)
                        # y_fit = slope * x_fit + intercept
                        # ax.plot(x_fit, y_fit, color="blue", linewidth=2, label="Best Fit")

                        # # Correlation
                        # pearson_corr, _ = pearsonr(x[mask], y[mask])
                        # spearman_corr, _ = spearmanr(x[mask], y[mask])
                        # ax.text(
                        #     0.5, 0.94,
                        #     f"Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}",
                        #     color="blue", fontsize=14, ha="center", va="top", transform=ax.transAxes
                        # )

                        # Plot the actual y=x line in blue color
                        ax.plot(
                            [bin_start, max_num_bound],
                            [bin_start, max_num_bound],
                            color="blue",
                            linestyle="--",
                            linewidth=2,
                            label="y=x"
                        )

                        # Set x and y tick label size larger
                        ax.tick_params(axis='x', labelsize=13)
                        ax.tick_params(axis='y', labelsize=13)

                        # Add cell line as y-axis label (bold, non-rotated) to the left of first column
                        if col_idx == 0:
                            ax.set_ylabel(f"{cell_line}", fontsize=20, fontweight="bold", rotation=0, labelpad=10, va='center', ha='right')
                        else:
                            ax.set_ylabel("")

                        # Add cutoff as column title (bold)
                        if row_idx == 0:
                            ax.set_title(f"{cutoff}", fontsize=22, fontweight="bold", pad=10)

                        # Only add colorbar to the right of the last subplot in the row
                        if col_idx == ncols - 1:
                            cbar = fig.colorbar(h, ax=ax, orientation="vertical", fraction=0.05, pad=0.04)
                            if row_idx == 0:
                                cbar.ax.set_title("Log\nCounts" if log_norm else "Counts", fontsize=12, pad=10)
                            cbar.ax.tick_params(labelsize=13)

                col_type_text = "Features" if col_type == "feature" else "RBPs"

                # Add meta x and y axis labels
                fig.supxlabel(f"# Bound {col_type_text}", fontsize=18)
                fig.supylabel(f"# Bound {col_type_text} w/ {self.latex_symbols['Unique-Binding']['local_SHAP']} > Cutoff", fontsize=17, x=0.01, y=0.42)
                fig.suptitle(
                    f"# Bound {col_type_text} vs # Bound {col_type_text} w/ {self.latex_symbols['Unique-Binding']['local_SHAP']} > Cutoff\n\nNOTE: using 'Unique Binding'\nNOTE 2: bins start at 1 and step by 2 for each x axis\nNOTE 3: colorbar is {'Logarithmic' if log_norm else 'Linear'} and shared across each row",
                    fontsize=12, y=1.02, 
                )

                # Add a single y=x legend to the plot at x=0.1 and y=0.45 by pulling the handle from the first subplot
                handles, labels = axes[0, 0].get_legend_handles_labels()
                idx = labels.index("y=x")
                fig.legend(
                    handles=[handles[idx]],
                    labels=["y=x"],
                    loc="center",
                    bbox_to_anchor=(0.08, 0.41),
                    fontsize=14,
                    frameon=True
                )

                plt.tight_layout()
                plt.savefig(self.FIGURES["per_row_num_bound_and_percent_greater_than_cutoff"]["num_bound_vs_num_bound_gt_cutoff"][col_type][f"{'log' if log_norm else 'linear'}"], dpi=600, bbox_inches='tight')
                plt.show()


    def assert_SHAP_additivity(self):

        data_file = self.CACHE_INFO["SHAP_additivity_assertions"]

        if Path(data_file).exists():

            logger.success(f"FROM CACHE: loading data for SHAP additivity assertions")

            additivity_df = pl.read_csv(data_file, separator="\t", dtypes={"model": str}).to_pandas()

            probability_df = additivity_df.copy()
            probability_df["Source"] = probability_df["model"].astype(str)

            # Set cell line and source order
            cell_line_order = sorted(probability_df["cell_line"].unique())
            source_order = [str(i) for i in range(5)] + ["Average"]

            plt.figure(figsize=(7, 4), dpi=300)
            
            ax = plt.gca()
            sns.violinplot(
                data=probability_df,
                x="cell_line",
                y="difference_probability",
                hue="Source",
                order=cell_line_order,
                hue_order=source_order,
                cut=0,
                scale="width",
                inner="box"
            )

            ax.set_title("Difference between Predicted PSI and\n[sum(Local SHAP) + Expected Value]", fontsize=10)
            ax.set_xlabel("Cell Line", fontsize=12)
            ax.set_ylabel("Difference (Probability)", fontsize=12)

            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            ax.legend(title="Source", bbox_to_anchor=(1.01, 0.7), loc="upper left", fontsize=10, title_fontsize=11)


            plt.tight_layout()
            plt.show()

        else: 
            logger.info(f"No cache found. Calculating SHAP additivity differences...")

            all_results = []
            # Load the final SHAP cache data as unique binding but as a lazyframe
            final_shap_lazy_dict = self.load_final_SHAP_data(underlying_data="Unique-Binding", as_lazyframe=True)

            for cell_line in self.cell_lines:
                # Get 5 SHAP lazyframes
                shap_lazyframes = self.get_SHAP_data_as_lazyframe(cell_line)
                schema = shap_lazyframes[0].collect_schema().names()
                binding_cols = [col for col in schema if col.endswith("_binding")]
                shap_cols = [col for col in schema if col.endswith("_shap")]

                # Get expected values for each model for this cell line
                expected_values = (
                    self.hash_metadata[
                        (self.hash_metadata['cell_line'] == cell_line) & 
                        (self.hash_metadata['name'] == "XGBRegressor")
                    ]
                    .sort_values("hash")
                    ['shap_expected_value']
                    .to_numpy()
                )

                indices_list = None
                for model_num, lf in enumerate(shap_lazyframes):
                    # Get expected value for this model
                    expected_value = expected_values[model_num]

                    # Create unique binding using binding columns for this model
                    unique_rows = (
                        lf
                        .unique(subset=binding_cols, maintain_order=True, keep="first")
                        .select(["index"] + shap_cols + ["Predictions"])
                        .sort('index')
                        .collect()
                    )
                    indices = unique_rows["index"].to_list()
                    if indices_list is None:
                        indices_list = indices
                    else:
                        assert indices_list == indices, "Indices are not aligned across models"

                    preds = unique_rows["Predictions"].to_numpy()

                    shap_sum = unique_rows.select(shap_cols).to_numpy().sum(axis=1)
                    diff = preds - (shap_sum + expected_value)

                    for idx, d in zip(indices, diff):
                        all_results.append({
                            "cell_line": cell_line,
                            "model": model_num,
                            "index": idx,
                            "difference_probability": d
                        })
                
                final_shap_lazy = final_shap_lazy_dict[cell_line]
                final_shap_df = final_shap_lazy.select(["index", "Averaged Prediction (Probability)"] + shap_cols).sort("index").collect()
                # Assert indices match
                assert indices_list == final_shap_df["index"].to_list(), "Indices in final SHAP cache do not match indices_list"
                
                # Calculate differences for the average model
                shap_sum = final_shap_df.select(shap_cols).to_numpy().sum(axis=1)
                avg_expected_value = np.mean(expected_values)

                # Assert that the lengths match before proceeding
                assert len(shap_sum) == len(final_shap_df["Averaged Prediction (Probability)"]), "Length mismatch between shap_sum and Averaged Prediction (Probability)"
                diff_avg = final_shap_df["Averaged Prediction (Probability)"].to_numpy() - (shap_sum + avg_expected_value)

                for idx, d in zip(indices_list, diff_avg):
                    all_results.append({
                        "cell_line": cell_line,
                        "model": "Average",
                        "index": idx,
                        "difference_probability": d
                    })

                del unique_rows, final_shap_df, shap_sum, diff_avg, indices_list
                gc.collect()

            # Save as DataFrame to data_file with gzip compression
            out_df = pd.DataFrame(all_results)
            out_df.to_csv(data_file, sep="\t", index=False, compression="gzip")


    def find_shap_waterfall_plot_examples(self): 

        if not hasattr(self, "final_all_data_SHAP_data"):
            self.load_final_SHAP_data(underlying_data="All-Data", as_lazyframe=False)
        
        PSI_THRESHOLDS = [0.1, 0.9]
        BINDING_SUM_THRESHOLD = 5
        PREDICTION_ERROR_THRESHOLD = 0.2

        # Efficiently collect unique RBP_KD_Target values for each cell line
        rbp_kd_targets = {
            cell_line: set(self.final_all_data_SHAP_data[cell_line].select("RBP_KD_Target").unique().get_column("RBP_KD_Target").to_list())
            for cell_line in self.cell_lines
        }

        # Intersect to get RBPs with knockdown in both cell lines, excluding "CTRL"
        rbps_both = rbp_kd_targets[self.cell_lines[0]].intersection(rbp_kd_targets[self.cell_lines[1]])
        rbps_both.discard("CTRL")

        rbps_both = sorted(rbps_both)

        # create lazy frames for each cell line with the base filtering conditions
        base_filtering = {}
        for cell_line in self.cell_lines:
            base_filtering[cell_line] = self.final_all_data_SHAP_data[cell_line].filter(
                (pl.col("has_RBP_KD") == True) &
                (pl.col("Partition") == "Test") &
                (pl.col("Target_PSI") < PSI_THRESHOLDS[1]) & 
                (pl.col("Target_PSI") > PSI_THRESHOLDS[0]) &
                (pl.col("Averaged Prediction (Probability)") < PSI_THRESHOLDS[1]) &
                (pl.col("Averaged Prediction (Probability)") > PSI_THRESHOLDS[0]) &
                ((pl.col("Target_PSI") - pl.col("Averaged Prediction (Probability)")).abs() < PREDICTION_ERROR_THRESHOLD) & 
                (pl.col("Binding Sum") > BINDING_SUM_THRESHOLD) &
                (pl.col("RBP_KD_Target").is_in(rbps_both)) 
            )

        # Collect indices for both cell lines matching RBP_KD_Target and Target_PSI thresholds
        indices_per_cell_line = {}
        for cell_line in self.cell_lines:
            indices_per_cell_line[cell_line] = set(
                base_filtering[cell_line].select("index").get_column("index").to_list()
            )

        # Find intersection of indices between both cell lines
        matching_indices = set.intersection(*indices_per_cell_line.values())
        logger.info(f"Found {len(matching_indices)} indices matching {len(rbps_both)} RBPs with knockdown in both cell lines, and Target_PSI in {PSI_THRESHOLDS}")

        # Now filter each cell line's DataFrame by the intersection of indices, overwriting base_filtering
        for cell_line in self.cell_lines:
            base_filtering[cell_line] = base_filtering[cell_line].filter(
                pl.col("index").is_in(matching_indices)
            )

        # Example: print number of filtered rows per cell line
        for cell_line, df in base_filtering.items():
            logger.info(f"{cell_line}: {df.shape[0]} rows with RBP_KD_Target in both cell lines, Target_PSI in {PSI_THRESHOLDS}, and matching indices")

        for cell_line in self.cell_lines:
            df = base_filtering[cell_line]
            
            shap_cols = [col for col in df.columns if col.endswith("_shap")]
            binding_cols = [col for col in df.columns if col.endswith("_binding")]
            remaining_cols = [col for col in df.columns if col not in shap_cols and col not in binding_cols]

            # Calculate average non-zeroness per row (mean absolute value across SHAP columns)
            avg_non_zeroness = df.select([pl.col(col).abs() for col in shap_cols]).sum_horizontal() / len(shap_cols)
            # Compute average signed SHAP per row (mean of SHAP columns)
            avg_row_shap = df.select([pl.col(col) for col in shap_cols]).sum_horizontal() / len(shap_cols)

            # Compute difference between avg_non_zeroness and abs(avg_row_shap)
            # This gives a signed difference where positive means avg_non_zeroness is greater than avg_row_shap
            diff_nonzeroness_signed = avg_non_zeroness - avg_row_shap.abs()

            # Add new columns and sort by diff_nonzeroness_signed
            filtered_df = df.with_columns([
                pl.Series("avg_non_zeroness", avg_non_zeroness),
                pl.Series("avg_row_shap", avg_row_shap),
                pl.Series("diff_nonzeroness_signed", diff_nonzeroness_signed)
            ]).sort("diff_nonzeroness_signed", descending=True)

            # Select the three new columns first, then all shap columns
            ordered_cols = ["avg_non_zeroness", "avg_row_shap", "diff_nonzeroness_signed"] + remaining_cols + binding_cols 
            filtered_df = filtered_df.select(ordered_cols)

            base_filtering[cell_line] = filtered_df
            
        del indices_per_cell_line, matching_indices,
        gc.collect()

        # Convert base_filtering to pandas DataFrames and save as a pickle file
        output_dict = {cell_line: base_filtering[cell_line].to_pandas() for cell_line in base_filtering}

        with open(self.CACHE_INFO["waterfall_plot_data"], "wb") as f:
            pickle.dump(output_dict, f)
        
        self.delete_data(data_type='Final SHAP Data All Data')



    def is_position_3_4_activating_and_others_repressing(self): 

        BOUND_LOCAL_SHAP = self.CACHE_INFO["position_3_4_activating_and_others_repressing"]["Bound Local SHAP Values"]
        NOT_BOUND_LOCAL_SHAP = self.CACHE_INFO["position_3_4_activating_and_others_repressing"]["NOT Bound Local SHAP Values"]

        if Path(BOUND_LOCAL_SHAP).exists() and Path(NOT_BOUND_LOCAL_SHAP).exists():

            logger.info("FROM CACHE: loading dataframes and plotting...")
            
            for file_path, col_name in [(BOUND_LOCAL_SHAP, "Bound Local SHAP"), (NOT_BOUND_LOCAL_SHAP, "NOT Bound Local SHAP")]:
                df = pl.read_csv(file_path, separator="\t").to_pandas()

                position_order = sorted(df["Position"].unique())
                position_group_order = sorted(df["Position Group"].unique())
                cell_line_order = sorted(df["Cell Line"].unique()) 

                latex_symbol_key = "local_SHAP_bound" if col_name == "Bound Local SHAP" else "local_SHAP_NOT_bound"
                latex_symbol = self.latex_symbols["Unique-Binding"][latex_symbol_key]

                # Helper color mapping (slightly lighter red/blue)
                def color_for_position(pos: int):
                    return "#ff9999" if pos in (3, 4) else "#99ccff"

                def color_for_group(group: str):
                    return "#ff9999" if group.strip() in ("3, 4", "3,4", "3 4") else "#99ccff"

                # Prepare figure (slightly increased wspace and tighter row spacing)
                fig = plt.figure(figsize=(14, 10), dpi=300)
                gs = gridspec.GridSpec(
                    2, 2,
                    width_ratios=[7, 3],
                    height_ratios=[1, 1],
                    wspace=0.08,
                    hspace=0.06
                )

                # Axes with shared x per column and shared y per row
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
                ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
                ax4 = fig.add_subplot(gs[1, 1], sharex=ax2, sharey=ax3)

                # ---------------- Violinplots using seaborn (color by position, hatch by cell line) ---------------- #
                # Ax1: by Position with hue=Cell Line
                sns.violinplot(
                    data=df,
                    x="Position",
                    y=col_name,
                    hue="Cell Line",
                    order=position_order,
                    hue_order=cell_line_order,
                    cut=0,
                    linewidth=1,
                    density_norm="width",
                    inner="box",
                    inner_kws={"color": "black"},  # ensure inner box is black
                    palette=["white", "white"],  # set neutral; recolor bodies below by position
                    ax=ax1
                )

                # Apply hatches and position-based facecolors to violins
                n_pos = len(position_order)
                n_hue = len(cell_line_order)
                coll_list = [c for c in ax1.collections if isinstance(c, mcoll.PolyCollection)]
                coll_list = coll_list[: n_pos * n_hue]  # only the violin bodies
                
                # annotate counts above each individual violin using its polygon center
                if len(coll_list) == n_pos * n_hue:
                    ymin, ymax = ax1.get_ylim()
                    new_ymax = ymax + 0.1 * (ymax - ymin)
                    ax1.set_ylim(ymin, new_ymax)
                    
                    text_y_label = new_ymax - (.1 * new_ymax)

                    for i_pos in range(n_pos):
                        for i_hue in range(n_hue):
                            idx = i_pos * n_hue + i_hue
                            coll = coll_list[idx]
                            # set facecolor by position (constant across cell lines)
                            coll.set_facecolor(color_for_position(position_order[i_pos]))
                            
                            # hatch mapping: HepG2 -> '', K562 -> 'xx'
                            if cell_line_order[i_hue] == "K562":
                                coll.set_hatch('xx')
                            else:
                                coll.set_hatch(None)

                            # edge color for visibility
                            coll.set_edgecolor('black')

                            # compute annotation position
                            try:
                                verts = coll.get_paths()[0].vertices
                                x_center = float(np.mean(verts[:, 0]))
                            except Exception:
                                # fallback if vertices not accessible
                                x_center = i_pos

                            # count points
                            n_points = int(
                                df[(df["Position"] == position_order[i_pos]) & (df["Cell Line"] == cell_line_order[i_hue])][col_name].dropna().shape[0]
                            )
                            ax1.text(
                                x_center,
                                text_y_label,
                                f"{n_points:.1e}",
                                ha="center",
                                va="bottom",
                                fontsize=6,
                                color="black"
                            )
                # remove per-subplot labels; shared labels handled later
                ax1.set_xlabel("")
                ax1.set_ylabel("")

                # Ax2: by Position Group with hue=Cell Line
                sns.violinplot(
                    data=df,
                    x="Position Group",
                    y=col_name,
                    hue="Cell Line",
                    order=position_group_order,
                    hue_order=cell_line_order,
                    cut=0,
                    linewidth=1,
                    density_norm="width",
                    inner="box",
                    inner_kws={"color": "black"},  # ensure inner box is black
                    palette=["white", "white"],  # set neutral; recolor bodies below by group
                    ax=ax2
                )
                coll_list2 = [c for c in ax2.collections if isinstance(c, mcoll.PolyCollection)]
                coll_list2 = coll_list2[: len(position_group_order) * n_hue]

                if len(coll_list2) == len(position_group_order) * n_hue:
                    ymin2, ymax2 = ax2.get_ylim()
                    new_ymax2 = ymax2 + 0.05 * (ymax2 - ymin2)
                    ax2.set_ylim(ymin2, new_ymax2)
                    # Define a constant y offset for annotation, similar to ax1
                    text_y_label2 = new_ymax2 - (0.17 * new_ymax2)

                    for i_grp in range(len(position_group_order)):
                        for i_hue in range(n_hue):
                            idx = i_grp * n_hue + i_hue
                            coll = coll_list2[idx]
                            # set facecolor by position group
                            coll.set_facecolor(color_for_group(position_group_order[i_grp]))
                            
                            if cell_line_order[i_hue] == "K562":
                                coll.set_hatch('xx')
                            else:
                                coll.set_hatch(None)

                            coll.set_edgecolor('black')
                            
                            try:
                                verts = coll.get_paths()[0].vertices
                                x_center = float(np.mean(verts[:, 0]))

                            except Exception:
                                x_center = i_grp

                            n_points = int(
                                df[(df["Position Group"] == position_group_order[i_grp]) & (df["Cell Line"] == cell_line_order[i_hue])][col_name].dropna().shape[0]
                            )
                            ax2.text(
                                x_center,
                                text_y_label2,
                                f"{n_points:.2e}",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                                color="black"
                            )
                ax2.set_xlabel("")
                ax2.set_ylabel("")

                # ---------------- Bar plots (custom) ---------------- #
                bar_data = df.groupby(["Position", "Cell Line"], as_index=False)[col_name].mean()
                bar_data_group = df.groupby(["Position Group", "Cell Line"], as_index=False)[col_name].mean()

                # Ax3: by Position
                x_pos = np.arange(len(position_order))
                bar_width = 0.4
                gap = 0.03  

                for ci, cell_line in enumerate(cell_line_order):
                    offset = (-bar_width / 2 - gap) if ci == 0 else (bar_width / 2 + gap)  # increased gap
                    hatch = None if cell_line == "HepG2" else "xx"
                    
                    for i, pos in enumerate(position_order):
                        
                        row = bar_data[(bar_data["Position"] == pos) & (bar_data["Cell Line"] == cell_line)]
                        
                        assert len(row)==1, f"Expected exactly 1 row for Position {pos} and Cell Line {cell_line}, but got {len(row)} rows."
                        val = row[col_name].values[0]

                        c = color_for_position(pos)
                        ax3.bar(
                            x_pos[i] + offset,
                            val,
                            width=bar_width,
                            color=c,
                            edgecolor="black",
                            linewidth=1.5,
                            hatch=hatch
                        )
                        
                        # Calculate a constant offset as 5% of the y-axis range for ax3 (like ax4)
                        ymin_ax3, ymax_ax3 = ax3.get_ylim()
                        y_offset = 0.05 * (ymax_ax3 - ymin_ax3)
                        # Annotation with increased distance, larger size, and 3f formatting, color by sign
                        if val >= 0:
                            ax3.text(
                                x_pos[i] + offset,
                                val + y_offset,
                                f"{val:.3f}",
                                ha="center",
                                va="bottom",
                                fontsize=11,
                                color="red"
                            )
                        else:
                            ax3.text(
                                x_pos[i] + offset,
                                val - y_offset,
                                f"{val:.3f}",
                                ha="center",
                                va="top",
                                fontsize=11,
                                color="blue"
                            )

                ax3.set_xticks(x_pos)
                ax3.set_xticklabels([str(p) for p in position_order], color="black")
                
                # Do not color x tick labels; always black
                for tick in ax3.get_xticklabels():
                    tick.set_color("black")

                ax3.set_xlabel("")  # shared label later
                ax3.set_ylabel("")  # shared label later

                # Ax4: by Position Group
                x_grp = np.arange(len(position_group_order))
                for ci, cell_line in enumerate(cell_line_order):

                    offset = (-bar_width / 2 - gap) if ci == 0 else (bar_width / 2 + gap)
                    hatch = None if cell_line == "HepG2" else "xx"

                    for i, group in enumerate(position_group_order):
                        row = bar_data_group[(bar_data_group["Position Group"] == group) & (bar_data_group["Cell Line"] == cell_line)]

                        val = row[col_name].values[0]
                        c = color_for_group(group)

                        ax4.bar(
                            x_grp[i] + offset,
                            val,
                            width=bar_width,
                            color=c,
                            edgecolor="black",
                            linewidth=1.5,
                            hatch=hatch
                        )

                        # Calculate a constant offset as 3% of the y-axis range for ax4
                        ymin_ax4, ymax_ax4 = ax4.get_ylim()
                        y_offset = 0.03 * (ymax_ax4 - ymin_ax4)

                        if val >= 0:
                            ax4.text(
                                x_grp[i] + offset,
                                val + y_offset,
                                f"{val:.3f}",
                                ha="center",
                                va="bottom",
                                fontsize=11,
                                color="red"
                            )
                        else:
                            ax4.text(
                                x_grp[i] + offset,
                                val - y_offset,
                                f"{val:.3f}",
                                ha="center",
                                va="top",
                                fontsize=11,
                                color="blue"
                            )

                ax4.set_xticks(x_grp)
                ax4.set_xticklabels(position_group_order, color="black")

                for tick in ax4.get_xticklabels():
                    tick.set_color("black")

                ax4.set_xlabel("")  # shared label later
                ax4.set_ylabel("")  # shared label later

                # Expand y-limits for bar plots (±10%)
                for ax in (ax3, ax4):
                    ymin_b, ymax_b = ax.get_ylim()
                    span_b = (ymax_b - ymin_b)
                    ax.set_ylim(ymin_b - 0.10 * span_b, ymax_b + 0.10 * span_b)

                # Horizontal y=0 line (all subplots): orange, dotted, thicker by 50%
                for ax in (ax1, ax2, ax3, ax4):
                    ax.axhline(0, color="orange", linestyle=":", linewidth=2, zorder=10)

                # Remove subplot titles (explicitly ensure none)
                for ax in (ax1, ax2, ax3, ax4):
                    ax.set_title("")

                # Shared labels: one y label per row, one x label per column
                ax1.set_ylabel(latex_symbol, fontsize=26)
                ax2.set_ylabel("")
                ax3.set_ylabel(f"Mean({latex_symbol})", fontsize=26)
                ax4.set_ylabel("")
                # Place y-axis labels further left and aligned at the same x coordinate
                shared_ylabel_x = -0.10
                ax1.yaxis.set_label_coords(shared_ylabel_x, 0.5)
                ax3.yaxis.set_label_coords(shared_ylabel_x, 0.5)
                
                ax3.set_xlabel("Position", fontsize=26, y=-0.01)
                ax4.set_xlabel("Position Group", fontsize=22, y=-0.01)

                # Ticks visibility rules:
                # - Keep tick markers on all subplots
                # - Remove y tick text labels for second column (ax2, ax4)
                # - Remove x tick text labels for first row (ax1, ax2)
                ax2.tick_params(labelleft=False)
                ax4.tick_params(labelleft=False)
                ax1.tick_params(labelbottom=False)
                ax2.tick_params(labelbottom=False)

                # Increase tick label sizes for all subplots
                for _ax in (ax1, ax2, ax3, ax4):
                    _ax.tick_params(axis='x', labelsize=22)
                    _ax.tick_params(axis='y', labelsize=18)

                # Remove legends inside subplots
                for ax in (ax1, ax2, ax3, ax4):
                    leg = ax.get_legend()
                    if leg:
                        leg.remove()
                    
                # Single, simplified legend outside (to the right):
                # White rectangles; HepG2 (no hatch), K562 ('xx' hatch)
                legend_handles = [
                    Patch(facecolor="white", edgecolor="black", hatch=None, label="HepG2"),
                    Patch(facecolor="white", edgecolor="black", hatch="xx", label="K562"),
                ]

                fig.legend(
                    handles=legend_handles,
                    labels=[h.get_label() for h in legend_handles],
                    loc="center left",
                    bbox_to_anchor=(0.9, 0.55),
                    fontsize=22,
                    title="Cell Line",
                    title_fontsize=26,
                    frameon=False
                )

                fig.suptitle(f"{latex_symbol}: Distributions by Position and Cell Line", fontsize=16, y=0.93)
                plt.tight_layout(rect=[0, 0, 0.85, 1])
                plt.savefig(self.FIGURES["position_3_4_activating_and_others_repressing"][col_name], dpi=600, bbox_inches='tight')
                plt.show()

        else: 
        
            logger.info("No cache found. Compiling dataframe with local SHAP values by binding value, per position, per cell line...")
            
            file_to_binding_value = {
                NOT_BOUND_LOCAL_SHAP: 0,
                BOUND_LOCAL_SHAP: 1
            }

            for key in file_to_binding_value.keys():

                local_shap = self.get_local_SHAP_based_on_binding_and_covariates(
                    binding_value=file_to_binding_value[key],
                    binding_pattern_type = "Unique-Binding",
                )

                if key == BOUND_LOCAL_SHAP:
                    col_name = "Bound Local SHAP"
                elif key == NOT_BOUND_LOCAL_SHAP:
                    col_name = "NOT Bound Local SHAP"
        
                # Accumulate all cell line/position/SHAP values into a single DataFrame
                all_rows = []
                for cell_line, shap_col, series in local_shap:
                    rbp , pos = self.get_RBP_position(shap_col)
                    grouped_positions = "3, 4" if pos in [3, 4] else "1, 2, 5, 6"

                    for val in series.to_numpy():
                        all_rows.append({
                            "Cell Line": cell_line,
                            "RBP": rbp,
                            "Position": pos,
                            "Position Group": grouped_positions,
                            col_name: val
                        })
                
                file_prefix = key.replace(".gz", "")

                # Save the entire data as a DataFrame for downstream usage (using polars for speed)
                pl.DataFrame(all_rows).with_columns(
                        pl.col("Position").cast(pl.UInt8)
                    ).sort(
                        ["Cell Line", "RBP", "Position"], 
                        maintain_order=True
                    ).write_csv(
                        file_prefix, 
                        separator="\t"
                    )
                
                os.system(f"gzip -f {file_prefix}")

                del local_shap, all_rows
                gc.collect()

    
    def compare_bound_global_SHAP_and_signed_mean_bound_local_SHAP(self):

        # Load Bound-Only Global SHAP and Signed Local SHAP Mean (Bound-Only)
        bound_global = self.calculate_specialized_global_SHAP(
            mode="Bound-Only",
            condition=None,
            underlying_data="Unique-Binding"
        )
        signed_mean_bound = self.calculate_specialized_global_SHAP(
            mode="Signed-Local-SHAP-Mean-Bound-Only",
            condition=None,
            underlying_data="Unique-Binding"
        )

        # Build long-form table with per-feature values per cell line
        rows = []
        for cell_line in self.cell_lines:
            x_df = bound_global[cell_line].copy()         # index: Position, columns: RBP
            y_df = signed_mean_bound[cell_line].copy()    # index: Position, columns: RBP

            for pos in x_df.index:
                for rbp in x_df.columns:
                    x_val = x_df.at[pos, rbp]
                    y_val = y_df.at[pos, rbp]

                    # Keep only rows where both are not NaN; take absolute value for signed mean
                    if pd.notna(x_val) and pd.notna(y_val):
                        rows.append({
                            "Cell Line": cell_line,
                            "RBP": rbp,
                            "Position": pos,
                            "Feature": f"{rbp}_{pos}",
                            "Bound Global SHAP": float(x_val),
                            "Signed Mean Bound Local SHAP (abs)": float(abs(y_val)),
                        })

        combined_df = pd.DataFrame(rows)

        # Plot: 2 columns (one per cell line)
        ncols = len(self.cell_lines)
        fig, axes = plt.subplots(1, ncols, figsize=(7, 4), dpi=300, sharex=True, sharey=True)

        x_label = self.latex_symbols["Unique-Binding"]["Bound-Only"]
        y_label = f"|{self.latex_symbols['Unique-Binding']['Signed-Local-SHAP-Mean-Bound-Only']}|"

        for ax, cell_line in zip(axes, self.cell_lines):
            plot_df = combined_df[combined_df["Cell Line"] == cell_line].copy()

            x = plot_df["Bound Global SHAP"].to_numpy()
            y = plot_df["Signed Mean Bound Local SHAP (abs)"].to_numpy()

            pearson_corr, spearman_corr = pearsonr(x, y)[0], spearmanr(x, y)[0]

            sns.scatterplot(
                x=x,
                y=y,
                ax=ax,
                color="deepskyblue",
                edgecolor="black",
                alpha=0.6,
                s=10
            )

            # y=x line
            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())
            ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1)

            ax.set_title(f"{cell_line}", fontsize=12)
            # Do not set individual axis labels since axes are shared
            ax.text(
                0.98, 0.02,
                f"Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}\nPoints: {len(plot_df)}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='bottom',
                horizontalalignment='right'
            )

        fig.suptitle("Bound-Only Global SHAP vs |Signed Mean Bound-Only Local SHAP|", fontsize=10, y=1.02)
        fig.supxlabel(x_label, fontsize=12)
        fig.supylabel(y_label, fontsize=12)
        plt.tight_layout()
        plt.show()


###############################################################
###############################################################
###############################################################
###############################################################
################ HACKY AND TEMPORARY FUNCTIONS ################
###############################################################
###############################################################
###############################################################
###############################################################

    def hacky_not_bound_global_SHAP_by_number_binding(self): 
        combos = [
            ("HepG2", "PRPF8_4"),
            ("HepG2", "BCLAF1_4"),
            ("K562", "AQR_4"),
        ]

        for cell_line, feature in combos:
            rbp, pos = feature.split("_")
            pos = int(pos)
            binding_col = f"{rbp}_{pos}_binding"
            shap_col = f"{rbp}_{pos}_shap"

            shap_lazyframes = self.get_SHAP_data_as_lazyframe(cell_line)

            # For each lazyframe, subset to rows where binding_col == 0, sort by 'index', and collect
            filtered_dfs = [
                lf.filter(pl.col(binding_col) == 0).select(["index", shap_col]).collect().sort("index")
                for lf in shap_lazyframes
            ]

            # Calculate the mean SHAP values across the 5 filtered DataFrames
            mean_shap_df = self.calculate_pointwise_SHAP_metric_per_cell_line(
                cell_line_shap=filtered_dfs,
                metric="mean"
            )

            # Take the first lazyframe, filter to binding_col == 0, sort by 'index', and collect index + all "_binding" columns
            schema = shap_lazyframes[0].collect_schema().names()
            binding_cols = [col for col in schema if col.endswith("_binding")]
            df_binding = (
                shap_lazyframes[0]
                .filter(pl.col(binding_col) == 0)
                .select(["index"] + binding_cols)
                .collect()
                .sort("index")
            )

            # Compute horizontal sum of all "_binding" columns per row
            df_binding = df_binding.with_columns(
                pl.sum_horizontal([pl.col(col) for col in binding_cols]).alias("binding_sum")
            )

            # Add the mean SHAP value for shap_col to df_binding (indices are aligned due to sorting)
            df_binding = df_binding.with_columns(
                pl.Series(f"{shap_col}_mean", mean_shap_df[shap_col])
            )

            print(f"\n{cell_line} - {feature}")
            for cutoff in [1, 2, 3, 4, 7, 10]:
                subset = df_binding.filter(pl.col("binding_sum") >= cutoff)
                abs_mean = subset[f"{shap_col}_mean"].abs().mean()
                print(f"Cutoff >= {cutoff}: abs mean {shap_col}_mean = {abs_mean}")

    
    def hacky_plot_bound_unbound_signed_local_SHAP_vs_PSI(self):

        # Define the cell line and feature combinations
        combos = [
            ("HepG2", "PRPF8_4"),
            ("HepG2", "BCLAF1_4"),
            ("K562", "AQR_4"),
        ]

        for cell_line, feature in combos:
            rbp, pos = feature.split("_")
            pos = int(pos)
            binding_col = f"{rbp}_{pos}_binding"
            shap_col = f"{rbp}_{pos}_shap"
            psi_col = "Target_PSI"

            # Get SHAP lazyframes for the cell line
            shap_lazyframes = self.get_SHAP_data_as_lazyframe(cell_line)

            for binding_value in [1, 0]:
                # For each lazyframe, filter to binding_col == binding_value, sort by 'index', and select index, psi_col, shap_col
                dfs = []
                for lf in shap_lazyframes:
                    df = (
                        lf.filter(pl.col(binding_col) == binding_value)
                        .select(["index", psi_col, shap_col])
                        .collect()
                        .sort("index")
                    )
                    dfs.append(df)

                # Calculate mean SHAP value across the 5 DataFrames
                mean_shap_df = self.calculate_pointwise_SHAP_metric_per_cell_line(
                    cell_line_shap=[df.select(["index", shap_col]) for df in dfs],
                    metric="mean"
                )
                mean_shap_df = mean_shap_df.rename({shap_col: f"{shap_col}_mean"})

                # Use the first dataframe as the base for index and psi
                base_df = dfs[0].select(["index", psi_col])
                base_df = base_df.with_columns(
                    pl.Series(f"{shap_col}_mean", mean_shap_df[f"{shap_col}_mean"])
                )
                merged = base_df

                # Prepare data for plotting
                y = merged["Target_PSI"].to_numpy()
                x = merged[f"{shap_col}_mean"].to_numpy()

                # Prepare mask for SHAP >= 1e-6
                mask = np.abs(x) >= 1e-6

                fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
                plot_titles = [
                    "All Data - Linear Colorbar",
                    "All Data - Log Colorbar",
                    "Abs(Local SHAP) ≥ 1e-6 - Linear Colorbar",
                    "Abs(Local SHAP) ≥ 1e-6 - Log Colorbar"
                ]
                # Use axes.flat for simple iteration over axes
                plot_configs = [
                    (False, None, plot_titles[0]),
                    (False, LogNorm(), plot_titles[1]),
                    (True, None, plot_titles[2]),
                    (True, LogNorm(), plot_titles[3]),
                ]

                for ax, (use_mask, norm, title) in zip(axes.flat, plot_configs):
                    if use_mask:
                        x_plot = x[mask]
                        y_plot = y[mask]
                    else:
                        x_plot = x
                        y_plot = y

                    hb = ax.hexbin(
                        x_plot, y_plot,
                        gridsize=70,
                        cmap='viridis',
                        norm=norm,
                        edgecolors='black',
                        mincnt=1, 
                        linewidths=0.2
                    )
                    ax.set_ylabel("Actual PSI", fontsize=10)
                    ax.set_xlabel("Local SHAP (Mean Across 5 Models)", fontsize=10)
                    ax.set_title(title, fontsize=12)
                    plt.colorbar(hb, ax=ax)

                plt.suptitle(
                    f"{cell_line} - {feature}: Actual PSI vs Local SHAP\nfor ALL rows where feature {'bound' if binding_value == 1 else 'not bound'} (aka. binding == {binding_value})",
                    fontsize=18, y=0.99
                )
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()

    
    def hacky_plot_bound_unbound_all_data_vs_unique_binding_global_SHAP(self): 
        # Define output file path
        output_file = "./unique_binding_bound_unbound_global_SHAP.pkl"
        if os.path.exists(output_file):
            # Load unique binding global SHAP (result)
            with open(output_file, "rb") as f:
                result = pickle.load(f)
            logger.success(f"FROM CACHE: Loaded unique binding global SHAP from {output_file}")

            # Load "All Data" bound and unbound global SHAP from cache
            all_data_bound = self.calculate_specialized_global_SHAP(mode="Bound-Only", condition=None)
            all_data_unbound = self.calculate_specialized_global_SHAP(mode="NOT-Bound-Only", condition=None)

            # Prepare figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=200)
            cell_lines = ["HepG2", "K562"]
            conditions = ["bound", "unbound"]

            for row_idx, cell_line in enumerate(cell_lines):
                for col_idx, condition in enumerate(conditions):
                    ax = axes[row_idx, col_idx]

                    # Get unique binding and all data DataFrames for this cell line and condition
                    unique_df = result[cell_line][condition]
                    if condition == "bound":
                        all_data_df = all_data_bound[cell_line]
                    elif condition == "unbound":
                        all_data_df = all_data_unbound[cell_line]

                    # Melt to long format for merging
                    unique_long = unique_df.reset_index().melt(id_vars=unique_df.index.name or "index", var_name="RBP", value_name="unique_val")
                    unique_long = unique_long.rename(columns={unique_df.index.name or "index": "Position"})
                    unique_long["Feature"] = unique_long["RBP"].astype(str) + "_" + unique_long["Position"].astype(str)
                    all_data_long = all_data_df.reset_index().melt(id_vars=all_data_df.index.name or "index", var_name="RBP", value_name="all_data_val")
                    all_data_long = all_data_long.rename(columns={all_data_df.index.name or "index": "Position"})
                    all_data_long["Feature"] = all_data_long["RBP"].astype(str) + "_" + all_data_long["Position"].astype(str)

                    # Assert that all unique values of Feature in all_data_long and unique_long are the same
                    assert set(all_data_long["Feature"].unique()) == set(unique_long["Feature"].unique()), "Unique values of Feature do not match between all_data_long and unique_long"
                    # Merge on Feature
                    merged = pd.merge(all_data_long[["Feature", "all_data_val"]], unique_long[["Feature", "unique_val"]], on="Feature", how="inner", validate="1:1")

                    # Assertions
                    if condition == "unbound":
                        assert not merged.isnull().values.any(), "Unbound: Null values found in merged table"
                    elif condition == "bound":
                        # For each row, either both columns are null or both are not null
                        both_null = merged["all_data_val"].isnull() & merged["unique_val"].isnull()
                        both_not_null = (~merged["all_data_val"].isnull()) & (~merged["unique_val"].isnull())
                        assert (both_null | both_not_null).all(), "Bound: There are rows where only one column is null"

                    # Remove rows where both are null (for plotting)
                    merged = merged.dropna(subset=["all_data_val", "unique_val"], how="all")

                    # Scatterplot
                    x = merged["all_data_val"]
                    y = merged["unique_val"]

                    pearson_corr, _ = pearsonr(x, y)
                    spearman_corr, _ = spearmanr(x, y)
                
                    num_points = len(merged)

                    sns.scatterplot(x=x, y=y, ax=ax, color="deepskyblue", edgecolor="black", alpha=0.7, s=10)
                    ax.plot([x.min(), x.max()], [x.min(), x.max()], color="lightgreen", linestyle="--", linewidth=1, label="y=x")

                    # Annotate top 10 points with the largest absolute difference between axes
                    merged["abs_diff"] = (merged["all_data_val"] - merged["unique_val"]).abs()
                    top_diff = merged.nlargest(10, "abs_diff")
                    for _, row in top_diff.iterrows():
                        ax.text(
                            row["all_data_val"],
                            row["unique_val"],
                            row["Feature"],
                            fontsize=6,
                            color="red",
                            alpha=0.8
                        )

                    ax.set_title(f"{cell_line} - {condition.capitalize()}", fontsize=14)
                    ax.set_xlabel("All Data Global SHAP", fontsize=12)
                    ax.set_ylabel("Unique Binding Global SHAP", fontsize=12)
                    ax.text(
                        0.98, 0.02,
                        f"Pearson: {pearson_corr:.5f}\nSpearman: {spearman_corr:.5f}\nPoints: {num_points}",
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment='bottom',
                        horizontalalignment='right'
                    )

            plt.suptitle("Global SHAP: All Data vs Unique Binding Patterns\n(Columns: Bound/Unbound, Rows: HepG2/K562)", fontsize=16)
            plt.tight_layout()
            plt.show()

        else: 
            result = {}
            for cell_line in self.cell_lines:
                logger.info(f"Processing cell line: {cell_line}")
                shap_lazyframes = self.get_SHAP_data_as_lazyframe(cell_line)

                unique_dfs = []
                for lf in shap_lazyframes:
                    schema = lf.collect_schema().names()
                    binding_cols = [col for col in schema if col.endswith("_binding")]
                    shap_cols = [col for col in schema if col.endswith("_shap")]
                    # Get unique rows by binding pattern
                    lf_unique = lf.unique(subset=binding_cols, maintain_order=True, keep="first")
                    lf_selected = lf_unique.select(shap_cols + binding_cols + ["index"])
                    df_collected = lf_selected.collect().sort("index")
                    unique_dfs.append(df_collected)

                assert all(df["index"].to_list() == unique_dfs[0]["index"].to_list() for df in unique_dfs), "All unique_dfs must have identical 'index' values in the same order"

                # For each shap_col, compute abs mean for bound and unbound using pointwise metric function in parallel
                feature_means = {"bound": {}, "unbound": {}}

                def compute_abs_mean_for_shap_col_binding(shap_col, binding_value):
                    binding_col = shap_col.replace("_shap", "_binding")
                    filtered_dfs = [
                        df.filter(pl.col(binding_col) == binding_value).select(["index", shap_col]).sort("index")
                        for df in unique_dfs
                    ]
                    mean_df = self.calculate_pointwise_SHAP_metric_per_cell_line(
                        cell_line_shap=filtered_dfs,
                        metric="mean"
                    )
                    abs_mean = mean_df[shap_col].abs().mean()
                    return shap_col, binding_value, abs_mean

                tasks = []
                for shap_col in shap_cols:
                    for binding_status, binding_value in [("bound", 1), ("unbound", 0)]:
                        tasks.append((shap_col, binding_status, binding_value))

                with concurrent.futures.ThreadPoolExecutor(max_workers=self.get_slurm_job_num_cpus()) as executor:
                    futures = {
                        executor.submit(compute_abs_mean_for_shap_col_binding, shap_col, binding_value): (shap_col, binding_status)
                        for shap_col, binding_status, binding_value in tasks
                    }
                    for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"{cell_line} shap_col"):
                        shap_col, binding_value, abs_mean = future.result()
                        _, binding_status = futures[future]
                        feature_means[binding_status][shap_col] = abs_mean

                # Convert to DataFrames and then to heatmaps
                for binding_status in ["bound", "unbound"]:
                    df = pd.DataFrame([feature_means[binding_status]])
                    heatmap = self.convert_RBP_position_to_2d_heatmap(df)
                    if cell_line not in result:
                        result[cell_line] = {}
                    result[cell_line][binding_status] = heatmap

                del unique_dfs
                gc.collect()

            with open(output_file, "wb") as f:
                pickle.dump(result, f)
            logger.success(f"Saved unique binding bound/unbound global SHAP to {output_file}")
            return result


    def hacky_explore_SHAP_in_PSI_units(self): 
        TMP_SHAP_PSI_UNITS_DIR = "~/july_21_SHAP_in_PSI_units/"

        chosen_models = {
            "K562": "1a5a622d0e4fb1bc7aeedf84cc41f73cd02c52b204046c305a4746fe7101c069", 
            "HepG2": "fdf52464ba1145bed424d92827c559d851550a0117d96a630474c08e558ac4fd"
        }
        
        for cell_line, hash in chosen_models.items(): 
            feather_file = f"{TMP_SHAP_PSI_UNITS_DIR}/SHAP/regular/normal/shap_values/{hash}.feather"

            df = pl.scan_ipc(feather_file)
            
            # Subset to unique binding patterns using columns ending in "_binding"
            schema = df.collect_schema().names()
            binding_cols = [col for col in schema if col.endswith("_binding")]
            shap_cols = [col for col in schema if col.endswith("_shap")]

            unique_df = df.unique(subset=binding_cols, maintain_order=True, keep="first").select(
                ["index"] + shap_cols + binding_cols
            ).sort("index").collect()

            abs_mean_shap_dict = {
                shap_col: unique_df
                    .filter(pl.col(shap_col.replace("_shap", "_binding")) == 1)[shap_col]
                    .abs()
                    .mean()
                for shap_col in shap_cols
            }

            abs_mean_shap = pl.DataFrame([abs_mean_shap_dict])

            heatmap_df = self.convert_RBP_position_to_2d_heatmap(abs_mean_shap)
            heatmap_df.to_csv(f"{cell_line}_SHAP_in_PSI_bound_global_SHAP.tsv", sep="\t")

            display(abs_mean_shap)

            del unique_df
            gc.collect()

            
        for cell_line in self.cell_lines:  
            heatmap_df = pd.read_csv(f"{cell_line}_SHAP_in_PSI_bound_global_SHAP.tsv", sep="\t", index_col=0)

            # Cluster columns (RBPs) using Ward linkage
            linkage = sch.linkage(heatmap_df.T.fillna(0), method="ward")
            dendro = sch.dendrogram(linkage, no_plot=True)
            ordered_cols = [heatmap_df.columns[i] for i in dendro["leaves"]]
            heatmap_df = heatmap_df[ordered_cols]

            plt.figure(figsize=(35,9), dpi=300)
            sns.heatmap(
                heatmap_df,
                cmap="Blues",
                linewidths=0.5,
                linecolor="gray",
                annot=True,
                fmt=".3f",
                annot_kws={"size": 12, "rotation": 90}
            )
            plt.gca().set_facecolor('black')
            plt.title(f"{cell_line}: Global SHAP in PSI Units\nNOTE: Yogi chose a random model per cell line\nNOTE 2: columns clustered with Ward", fontsize=16)
            plt.xlabel("RBP")
            plt.ylabel("Position")
            plt.tight_layout()
            plt.show()


        # Plot both cell lines and both scales (linear/log) in a single figure
        fig, axes = plt.subplots(len(self.cell_lines), 2, figsize=(10, 9), dpi=300)
        for row_idx, cell_line in enumerate(self.cell_lines):
            
            # Load data for this cell line
            heatmap_df = pd.read_csv(f"{cell_line}_SHAP_in_PSI_bound_global_SHAP.tsv", sep="\t", index_col=0)
            bound_global_shap = self.calculate_specialized_global_SHAP(mode="Bound-Only", condition=None, underlying_data="Unique-Binding")[cell_line]
            # Assert RBPs and positions match
            assert set(heatmap_df.columns) == set(bound_global_shap.columns), "RBP columns mismatch"
            assert set(heatmap_df.index) == set(bound_global_shap.index), "Position index mismatch"
            
            # Flatten to feature arrays
            x_vals = []
            y_vals = []
            for pos in heatmap_df.index:
                for rbp in heatmap_df.columns:
                    x_vals.append(bound_global_shap.at[pos, rbp])
                    y_vals.append(heatmap_df.at[pos, rbp])
            
            x_vals = np.array(x_vals)
            y_vals = np.array(y_vals)
            
            # Mask for valid values
            mask = ~np.isnan(x_vals) & ~np.isnan(y_vals)
            x_valid = x_vals[mask]
            y_valid = y_vals[mask]
            
            # Mask for log scale
            mask_log = (x_valid > 0) & (y_valid > 0)
            
            # Iterate over columns: 0=linear, 1=log
            for col_idx in range(2):
                ax = axes[row_idx, col_idx]
                if col_idx == 0:
                    # Linear scale
                    x_plot = x_valid
                    y_plot = y_valid
                    pearson_corr, _ = pearsonr(x_plot, y_plot)
                    spearman_corr, _ = spearmanr(x_plot, y_plot)
                    n_points = len(x_plot)
                    ax.scatter(x_plot, y_plot, color="deepskyblue", edgecolor="black", alpha=0.4, s=10)
                    ax.set_title(f"{cell_line} - Linear Scale", fontsize=12)

                    # Label the top 10 points on each axis
                    top_10_x = np.argsort(x_plot)[-10:]
                    top_10_y = np.argsort(y_plot)[-10:]
                    labeled = set()
                    for idx in np.unique(np.concatenate([top_10_x, top_10_y])):
                        ax.text(
                            x_plot[idx]-0.05,
                            y_plot[idx]+0.01,
                            f"{heatmap_df.columns[idx % len(heatmap_df.columns)]}_{heatmap_df.index[idx // len(heatmap_df.columns)]}",
                            fontsize=4,
                            color="black",
                            alpha=0.8
                        )

                else:
                    # Log-log scale
                    x_plot = np.log10(x_valid[mask_log])
                    y_plot = np.log10(y_valid[mask_log])
                    pearson_corr, _ = pearsonr(x_plot, y_plot)
                    spearman_corr, _ = spearmanr(x_plot, y_plot)
                    n_points = mask_log.sum()
                    ax.scatter(x_plot, y_plot, color="deepskyblue", edgecolor="black", alpha=0.4, s=10)
                    ax.set_title(f"{cell_line} - Log-Log Scale", fontsize=12)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.text(
                    0.02, 0.98,
                    f"Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}\nPoints (Features): {n_points}",
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left'
                )
        plt.suptitle("PSI Units Global SHAP vs Log-Odds Bound Global SHAP\n\nNOTE: Using unique-binding\nNOTE 2: Comparing 'Averaged' Log-Odds SHAP w/ SINGLE Model PSI SHAP", fontsize=12, y=1.02)
        fig.supxlabel("Log-Odds Bound Global SHAP", fontsize=12)
        fig.supylabel("PSI Units Bound Global SHAP", fontsize=12)
        plt.tight_layout()
        plt.show()
    

    def hacky_check_shap_additivity(self): 

        # Get all model hashes for XGBRegressor from self.hash_metadata, filtering out hashes that begin with "8f31"
        # Sort by cell_line so hashes are grouped by cell line in the plot
        hashes = (
            self.hash_metadata[self.hash_metadata["name"] == "XGBRegressor"]
            .sort_values("cell_line")["hash"].tolist()
        )
        all_diffs = []
        all_labels = []

        for model_hash in hashes:
            shap_file = os.path.join(self.SHAP_DIR, f"{model_hash}.feather")
            lf = pl.scan_ipc(shap_file)
            schema = lf.collect_schema().names()

            shap_cols = [col for col in schema if col.endswith("_shap")]
            binding_cols = [col for col in schema if col.endswith("_binding")]
            pred_col = "Predictions"

            df = lf.unique(subset=binding_cols, maintain_order=True, keep="first").select(
                ["index"] + shap_cols + [pred_col]
            ).sort("index").collect()

            assert df.null_count().sum_horizontal().item() == 0, "Null values found in df"

            model_file = os.path.join(self.SHAP_MODEL_PICKLE_DIR, f"{model_hash}.pkl")
            with open(model_file, "rb") as f:
                model = pickle.load(f)
            expected_value = model.expected_value

            shap_sum = df.select(shap_cols).to_numpy().sum(axis=1)
            pred = df[pred_col].to_numpy()
            diff = pred - (shap_sum + expected_value)

            all_diffs.extend(diff)
            all_labels.extend([model_hash] * len(diff))

        # Prepare DataFrame for seaborn violinplot
        plot_df = pl.DataFrame({
            "Model": all_labels,
            "Diff": all_diffs
        })
        # Save as TSV using polars
        tsv_path = "model_diff_WITH_expected_value.tsv.gz"
        plot_df.write_csv(tsv_path, separator="\t", compression="gzip")

        diff_without_expected_value = pl.read_csv(
            "model_diff_WITHOUT_expected_value.tsv.gz", 
            separator="\t"
        ).to_pandas()

        diff_with_expected_value = pl.read_csv(
            "model_diff_with_expected_value.tsv.gz", 
            separator="\t"
        ).to_pandas()

        fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=300, sharex=True, sharey=True)
        datasets = [
            (diff_with_expected_value, r"$\mathrm{Prediction} - \left(\sum \mathrm{SHAP} + \mathrm{Expected\ Value}\right)$"),
            (diff_without_expected_value, r"$\mathrm{Prediction} - \sum \mathrm{SHAP}$"),
        ]
        for i, (data, title) in enumerate(datasets):
            # Add abbreviated model hash column
            data = data.copy()
            data["Model Hash (Abbreviated)"] = data["Model"].astype(str).str[:5]
            sns.violinplot(
                x="Model Hash (Abbreviated)", y="Diff", data=data, inner="box", color="skyblue", cut=0, density_norm='width', ax=axes[i]
            )
            axes[i].set_title(title, fontsize=13)

            axes[i].axhline(0, color="red", linestyle="--", linewidth=0.5)
            axes[i].axvline(4.5, color="green", linestyle="--", linewidth=1)

            # Add text annotation to subplot
            axes[i].text(
                2, -1,
                f"HepG2",
                fontsize=10,
                color="green",
                ha="center",
                va="top"
            )
            # Add text annotation to subplot
            axes[i].text(
                7, -1,
                f"K562",
                fontsize=10,
                color="green",
                ha="center",
                va="top"
            )

            axes[i].set_xlabel("")
            axes[i].set_ylabel("")

            axes[i].tick_params(axis='x', labelrotation=45, labelsize=8, labelbottom=True)
            axes[i].tick_params(axis='y', labelsize=10)

            # Add annotation above each violinplot: % of absolute values > 0.05
            model_hash_abbr_order = data["Model Hash (Abbreviated)"].unique()
            for j, model_hash_abbr in enumerate(model_hash_abbr_order):
                vals = data[data["Model Hash (Abbreviated)"] == model_hash_abbr]["Diff"].to_numpy()
                pct_gt_005 = (np.abs(vals) > 0.05).mean() * 100
                axes[i].text(
                    j-0.1, -0.2,
                    f"{pct_gt_005:.0f}%\nDiff > 0.05",
                    ha="center", va="top", fontsize=7, color="#fc6f03", zorder=10,
                )
        fig.supxlabel("Model Hash (Abbreviated)", fontsize=14, y=0.01)
        fig.supylabel("Difference", fontsize=14, x=0.01)
        plt.tight_layout()
        plt.show()


    def tmp(self, rbp=None): 
        
        bound_global_SHAP = self.calculate_specialized_global_SHAP(
            mode="Bound-Only", 
            underlying_data="Unique-Binding"
        )
        
        # For the given rbp, build a DataFrame: rows = cell lines, columns = positions (1-6), values = global SHAP
        df = pd.DataFrame(
            {cell_line: bound_global_SHAP[cell_line][rbp] for cell_line in self.cell_lines}
        ).T

        plt.figure(figsize=(8, 3))
        sns.heatmap(df, annot=True, fmt=".4f", cmap="Blues")
        plt.title(f"{rbp}: Bound-Only Global SHAP Across Cell Lines and Positions")
        plt.xlabel("Position")
        plt.ylabel("Cell Line")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    # Set up loguru logger: all output to stdout, nothing to stderr
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    parser = argparse.ArgumentParser(description="SHAP Network Investigator")

    PARALLELIZE_CHOICES = [
        "data_qc_assertions", 
        "shap_additivity_assertions", 
        "no_shap_variance_per_binding_pattern", 
        "create_final_shap_cache", 
        "global_SHAP_5_dfs_all_data", 
        "local_SHAP_mean_vs_variance_plot",
        "calculate_local_SHAP_percent_non_zero",
        "calculate_local_SHAP_mean_vs_variance_deciles_bound",
        "calculate_local_SHAP_mean_vs_variance_deciles_unbound", 
        "calculate_percent_positive_and_negative_local_SHAP_per_feature", 
        "pct_pos_neg_local_SHAP_per_feature_bound",
        "pct_pos_neg_local_SHAP_per_feature_unbound", 
        "arbs_narbs_bound", 
        "arbs_narbs_unbound", 
        "is_position_3_4_activating_and_others_repressing", 
        "signed_local_SHAP_mean_bound",
        "signed_local_SHAP_mean_unbound",
    ]

    parser.add_argument(
        "--parallelize",
        type=str,
        choices=PARALLELIZE_CHOICES,
        help="Specify the job type to parallelize. If provided, will run the job in parallel using sbatch.",
        required=False
    )

    parser.add_argument(
        "--job_type", 
        type=str,
        choices=PARALLELIZE_CHOICES, 
        required=False, 
        help="Specify the job type for specific job."
    )

    args = parser.parse_args()

    if args.parallelize:

        sbatch_prefix = "sbatch -N2 --partition=parallel -n32 --mem=256GB --account=platiglab"
        sbatch_command = f"{sbatch_prefix} --job-name={args.parallelize} --output=../SLURM_logs/{args.parallelize}.out --error=../SLURM_logs/{args.parallelize}.err --wrap='python3.11 {__file__} --job_type {args.parallelize}'"
        
        logger.info(f"Submitting job with sbatch command:\n\n{sbatch_command}")
        os.system(sbatch_command)
    
    elif args.job_type: 

        analyzer = ShapNetworkInvestigator()

        if args.job_type == "data_qc_assertions":
            analyzer.run_data_quality_assertions()
        
        elif args.job_type == "shap_additivity_assertions":
            analyzer.assert_SHAP_additivity()

        elif args.job_type == "no_shap_variance_per_binding_pattern":
            analyzer.check_no_SHAP_variance_per_binding_pattern()
        
        elif args.job_type == "create_final_shap_cache":
            analyzer.load_final_SHAP_data(underlying_data="All-Data")
        
        elif args.job_type == "global_SHAP_5_dfs_all_data":
            analyzer.calculate_global_SHAP(mode="5_dfs", binding_unique="All-Data")

        elif args.job_type == "local_SHAP_mean_vs_variance_plot":
            analyzer.plot_local_SHAP_mean_vs_variance()

        elif args.job_type == "calculate_local_SHAP_percent_non_zero": 
            analyzer.calculate_local_SHAP_percent_non_zero(mode="Unique-Binding")

        elif args.job_type.startswith("calculate_local_SHAP_mean_vs_variance_deciles"): 
            analyzer.calculate_local_SHAP_mean_vs_variance_deciles(
                mode="Bound-Only" if args.job_type.endswith("_bound") else "NOT-Bound-Only",
            )
        
        elif args.job_type.startswith("pct_pos_neg_local_SHAP_per_feature"): 
            analyzer.calculate_percent_positive_and_negative_local_SHAP_per_feature(
                underlying_data="Unique-Binding",
                mode="Bound-Only" if args.job_type.endswith("_bound") else "NOT-Bound-Only"
            )

        elif args.job_type.startswith("arbs_narbs_"):
            analyzer.calculate_activator_repressor_behavior_score(
                underlying_data="Unique-Binding",
                binding_mode = "Bound-Only" if args.job_type.endswith("_bound") else "NOT-Bound-Only",
            )

        elif args.job_type == "is_position_3_4_activating_and_others_repressing": 
            analyzer.is_position_3_4_activating_and_others_repressing()

        elif args.job_type.startswith("signed_local_SHAP_mean_"): 
            
            if args.job_type.endswith("_bound"):
                mode = "Signed-Local-SHAP-Mean-Bound-Only"
            elif args.job_type.endswith("_unbound"):
                mode = "Signed-Local-SHAP-Mean-NOT-Bound-Only"
            else:
                raise ValueError(f"Unknown job type: {args.job_type}")

            analyzer.calculate_specialized_global_SHAP(
                mode=mode, 
                condition=None,
                underlying_data="Unique-Binding"
            )
            
        else:
            raise ValueError(f"Unknown job type: {args.job_type}")