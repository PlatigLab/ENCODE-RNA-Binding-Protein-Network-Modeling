import polars as pl, pandas as pd, seaborn as sns, matplotlib.pyplot as plt, numpy as np, xgboost as xgb
import pathlib, glob, wandb, pickle, os, shap
from matplotlib.colors import LogNorm

from dataclasses import dataclass
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score

_=pl.Config.set_tbl_cols(100000)
_=pl.Config.set_tbl_rows(10000)
_=pl.Config.set_tbl_width_chars(10000)
_=pl.Config.set_fmt_str_lengths(10000)


@dataclass
class YogiBindingPatternAnalyzer:

    DATA_PATH = "/project/PlatigLab/data/RBP_ML/5_yogi_dataset_feb_2026_GENCODE_v24_v29_matching_exons"
    CACHE_DIR = "../__featherv2-cache__/"
    WANDB_DIR = "../outputs/wandb_results/"

    cell_lines = ["HepG2", "K562"]
    distance = 100
    binding_mode = "binary" 
    event_filter = "all-events"

    psi_bins = [0.1, 0.9]

    train_set = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9', 'chr11', 'chr13', 'chr15', 'chr17', 'chr19', 'chr21']
    validate_set = ['chr4', 'chr6', 'chr10', 'chr14', 'chr18', 'chr22']
    test_set = ["chr2", "chr8", "chr12", "chr16", "chr20"]

    random_seed = 17


    def __post_init__(self): 
        assert self.binding_mode =='binary'
        assert self.event_filter == 'all-events'
        assert self.distance == 100, "Distance must be 100"


    def read_data(self):

        modeling_input_data = {}
        binding_cols = {}    

        if len(glob.glob(f"{self.CACHE_DIR}/*.feather")) == len(self.cell_lines):  
            
            logger.info(f"FROM CACHE: loading all input binding data...")

            for cell_line in self.cell_lines:
                tmp_df = pl.read_ipc(f"{self.CACHE_DIR}/{cell_line}_{self.distance}.feather")

                modeling_input_data[cell_line] = tmp_df
                binding_cols[cell_line] = [col for col in tmp_df.columns if col.endswith("_binding")]

                logger.success(f"{cell_line} data loaded. Shape -- {tmp_df.shape}")

            logger.success("All data loaded from cache successfully.")

            self.modeling_input_data = modeling_input_data
            self.binding_cols = binding_cols
            
        else: 

            for cell_line in self.cell_lines:
                logger.info(f"Creating {self.binding_mode} binding data for {cell_line}...")

                binding_data = pl.scan_csv(f"{self.DATA_PATH}/{cell_line}_{self.distance}_{self.event_filter}_num-peaks-no-kd.tsv.gz", separator='\t')
                binding_cols = [col for col in binding_data.collect_schema().names() if col.endswith("_binding")]

                binding_data = binding_data \
                    .filter(pl.col("Total Read Counts") >= 40) \
                    .collect() 

                assert binding_data["chr"].is_in([f"chr{i}" for i in range(1, 23)]).all(), "The 'chr' column contains unexpected values"
                assert binding_data["index"].n_unique() == binding_data.shape[0], "The 'index' column contains duplicate values"
                assert self.binding_mode == "binary"

                binding_data = binding_data.with_columns([
                    pl.when(pl.col(col) > 1).then(1).otherwise(pl.col(col)).alias(col) 
                    for col in binding_cols
                ])
                assert all(binding_data[col].max() <= 1 for col in binding_cols), "Some values in binding columns are greater than 1"

                original_binding_data_shape = binding_data.shape

                unique_rbp_kd_targets = sorted(binding_data["RBP_KD_Target"].unique().to_list())
                modified_dfs = []
                for rbp_kd_target in unique_rbp_kd_targets:
                    subset_df = binding_data.filter(pl.col("RBP_KD_Target") == rbp_kd_target)

                    if rbp_kd_target != "CTRL":
                        binding_cols_to_zero = [col for col in binding_cols if col.startswith(f"{rbp_kd_target}_")]
                        assert len(binding_cols_to_zero) ==6, print(binding_cols_to_zero)
                        
                        subset_df = subset_df.with_columns([
                            pl.lit(0).alias(col) for col in binding_cols_to_zero
                        ])

                    modified_dfs.append(subset_df)

                binding_data = pl.concat(modified_dfs, how='vertical_relaxed').with_columns(
                    [pl.col(col).cast(pl.UInt8) for col in binding_cols]
                )

                assert binding_data.shape == original_binding_data_shape, "Dataframe shape changed after modification"
                binding_data = binding_data.sort("index")
                
                self._cache_to_featherv2(binding_data, f"{self.CACHE_DIR}/{cell_line}_{self.distance}.feather")
                logger.success(f"{cell_line} data created and cached. Shape -- {binding_data.shape}")
                
            logger.success("All data cached successfully.")

    
    def _cache_to_featherv2(self, df, output_path):
        
        logger.info(f"Caching to feather v2 @ {output_path}")

        output_path = pathlib.Path(output_path)
        assert not output_path.exists(), f"File already exists: {output_path}"

        df.write_ipc(
            output_path,
            compression='lz4',
        )

        logger.success(f"Data cached successfully @ {output_path}")

    
    def plot_psi_distribution_per_cell_line(self):
        fig, axes = plt.subplots(len(self.cell_lines), 1, figsize=(7, 5), dpi=300, sharex=True, sharey=True)

        for ax, cell_line in zip(axes, self.cell_lines):
            psi_data = self.modeling_input_data[cell_line].select("Target_PSI").to_pandas()
            
            total_points = psi_data.shape[0]
            total_points_millions = total_points / 1_000_000

            sns.histplot(
                data=psi_data,
                x="Target_PSI",
                bins=100,
                color='#87CEEB',  # Slightly less dark blue
                edgecolor='black',
                linewidth=0.8,  # Thicker edge width
                ax=ax,
                stat="percent"  # Show percentage instead of raw counts
            )


            ax.set_title(f"{cell_line}: PSI Distribution", fontsize=14)
            ax.set_xlabel("PSI", fontsize=12)
            ax.set_ylabel("", fontsize=12)

            # Add total points in a box at the top center
            ax.text(
                0.5, 0.9, f"# Graphs: {total_points_millions:.2f} million",
                horizontalalignment='center',
                verticalalignment='top',
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
            )
        
        fig.supylabel("% of Data in Bin", fontsize=16)
        plt.tight_layout()
        plt.show()


    def get_num_cpus(self):
        num_cpus = os.getenv('SLURM_CPUS_PER_TASK')
        if num_cpus is not None:
            return int(num_cpus)
        else:
            logger.warning("SLURM_CPUS_PER_TASK environment variable not set. Defaulting to 1 CPU.")
            return 1


    def get_rbp_and_position(self, string): 
        assert string.endswith("_binding")
        return string.split("_")[0], string.split("_")[1]
    

    def summate_binding_amount(self, df, total_binding=None): 
        
        assert total_binding is not None, "total_binding must be provided"

        binding_cols = [col for col in df.columns if col.endswith("_binding")]
        df = df.select(binding_cols)
        
        if total_binding: 
            num_rows = df.shape[0]

        df = df.sum()

        if total_binding:
            df = (df / num_rows)*100

        return df
        

    def convert_features_to_2d_table(self, df, column_normalized=None):

        assert column_normalized is not None, "Column normalized must be provided" 
        assert df.shape[0] == 1, "DataFrame must have exactly one row"
        assert all(col.endswith("_binding") for col in df.columns), "All columns must end with '_binding'"

        if isinstance(df, pl.DataFrame):
            df_pandas = df.to_pandas()
        else:
            df_pandas = df

        binding_cols = [col for col in df_pandas.columns if col.endswith("_binding")]
        rbp_positions = [self.get_rbp_and_position(col) for col in binding_cols]

        heatmap_data = {}
        for rbp, position in rbp_positions:
            if position not in heatmap_data:
                heatmap_data[position] = {}

            heatmap_data[position][rbp] = df_pandas[f"{rbp}_{position}_binding"].iloc[0]

        heatmap_df = pd.DataFrame.from_dict(heatmap_data, orient='index')
        assert not heatmap_df.isnull().values.any(), "There are missing or null values in the heatmap DataFrame"

        if column_normalized:
            heatmap_df = heatmap_df.div(heatmap_df.sum(axis=0), axis=1) * 100

            if heatmap_df.isnull().values.any():
                null_columns = heatmap_df.columns[heatmap_df.isnull().any()].tolist()
                logger.warning(f"NULLS DETECTED when column normalizing: {null_columns}")

        heatmap_df = heatmap_df.sort_index(axis=0).sort_index(axis=1)
        return heatmap_df
    

    def return_ward_hierarchical_clustering_order(self, df): 

        assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
        assert df.shape[1] > df.shape[0], "DataFrame must have more columns than rows for clustering"
        assert list(df.index) == list(map(str, range(1, 7))), "Rows of the DataFrame are not in the order of 1 to 6"
        

        return df.iloc[ 
                :,
                leaves_list(
                    linkage(df.T, method='ward')
                )
            ]



    def plot_entire_dataset_RBP_heatmaps(self): 
        
        for cell_line in self.cell_lines:
            for column_normalized in [False, True]:

                plotting_data = self.convert_features_to_2d_table(
                    self.summate_binding_amount(
                        self.modeling_input_data[cell_line],
                        total_binding=True
                    ),
                    column_normalized=column_normalized
                )

                plotting_data = plotting_data.dropna(axis=1, how='all')
                assert not plotting_data.isnull().values.any(), "plotting_data contains null values"
                assert plotting_data.max().max() <= 100, "Some values in the heatmap are greater than 1"

                plotting_data = self.return_ward_hierarchical_clustering_order(plotting_data)

                plt.figure(figsize=(35, 7), dpi=300)

                ax = sns.heatmap(
                    plotting_data,
                    cmap="Blues",
                    mask=plotting_data.isnull(),
                    cbar=True,
                    linewidths=.5,
                    linecolor='black',
                )
                ax.set_facecolor('gray')

                ax.tick_params(axis='x', which='major', labelsize=12, length=6, width=1.5)
                ax.tick_params(axis='y', which='major', labelsize=18, length=6, width=1.5)

                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=16)
                cbar.ax.set_position([0.76, 0.15, 0.04, 0.7])
                cbar.ax.set_title("%", fontsize=20)
                
                if column_normalized:
                    normalization_suffix = "Column Normalized"
                    figure_aim = "Positional Preference of RBPs"
                    normalization_filename_suffix = "column-normalized"
                    figure_aim_suffix = "positional-preference-of-rbps"
                else:
                    normalization_suffix = "NOT column normalized"
                    figure_aim = "% Graphs Bound"
                    normalization_filename_suffix = "not-column-normalized"
                    figure_aim_suffix = "percent-graphs-bound"

                plt.suptitle(f"{cell_line}: {figure_aim}\n{normalization_suffix}\nNOTE: hierarchically clustered w/ Ward", fontsize=26, x=0.45, y=1.08)
                # ax.set_title(f"{normalization_suffix}", fontsize=18, y=1.04)

                plt.savefig(f"../outputs/rbp_binding_heatmaps/all_data_heatmaps/{cell_line}_{figure_aim_suffix}_AKA_{normalization_filename_suffix}.png", bbox_inches='tight', dpi=300)
                plt.show()


    def stratify_data_by_PSI_bins(self): 

        logger.info("Stratifying data by PSI bins...")

        psi_bins_dict = {}

        for cell_line in self.cell_lines:
            psi_bins_dict[cell_line] = {
                f"PSI < {self.psi_bins[0]}": self.modeling_input_data[cell_line].filter(pl.col("Target_PSI") < self.psi_bins[0]),
                f"{self.psi_bins[0]} <= PSI <= {self.psi_bins[1]}": self.modeling_input_data[cell_line].filter((pl.col("Target_PSI") >= self.psi_bins[0]) & (pl.col("Target_PSI") <= self.psi_bins[1])),
                f"PSI > {self.psi_bins[1]}": self.modeling_input_data[cell_line].filter(pl.col("Target_PSI") > self.psi_bins[1])
            }

            for psi_bin, df in psi_bins_dict[cell_line].items():
                logger.info(f"{cell_line} - PSI bin '{psi_bin}': Shape -- {df.shape}")

        logger.success("Data stratified by PSI bins successfully.")
        self.psi_bins_dict = psi_bins_dict


    def plot_PSI_bin_stratified_RBP_heatmaps(self): 

        if not hasattr(self, "psi_bins_dict"): 
            self.stratify_data_by_PSI_bins()
        
        for cell_line in self.psi_bins_dict: 
            for bin in self.psi_bins_dict[cell_line]:
                for column_normalized in [False, True]:

                    plotting_data = self.convert_features_to_2d_table(
                        self.summate_binding_amount(
                            self.psi_bins_dict[cell_line][bin],
                            total_binding=True
                        ),
                        column_normalized=column_normalized
                    )
                    
                    plotting_data = plotting_data.dropna(axis=1, how='all')
                    assert not plotting_data.isnull().values.any(), "plotting_data contains null values"
                    assert plotting_data.max().max() <= 100, "Some values in the heatmap are greater than 1"

                    plotting_data = self.return_ward_hierarchical_clustering_order(plotting_data)

                    plt.figure(figsize=(35, 5), dpi=300)

                    ax = sns.heatmap(
                        plotting_data,
                        cmap="Blues",
                        mask=plotting_data.isnull(),
                        cbar=True,
                        linewidths=.5,
                        linecolor='black',
                    )
                    ax.set_facecolor('gray')

                    ax.tick_params(axis='x', which='major', labelsize=12, length=6, width=1.5)
                    ax.tick_params(axis='y', which='major', labelsize=18, length=6, width=1.5)

                    cbar = ax.collections[0].colorbar
                    cbar.ax.tick_params(labelsize=16)
                    cbar.ax.set_position([0.76, 0.15, 0.04, 0.7])
                    cbar.ax.set_title("%", fontsize=20)

                    if column_normalized:
                        normalization_suffix = "Column Normalized"
                        figure_aim = "Positional Preference of RBPs"
                        normalization_filename_suffix = "column-normalized"
                        figure_aim_suffix = "positional-preference-of-rbps"
                    else:
                        normalization_suffix = "NOT column normalized"
                        figure_aim = "% Graphs Bound"
                        normalization_filename_suffix = "not-column-normalized"
                        figure_aim_suffix = "percent-graphs-bound"

                    plt.suptitle(f"{cell_line} {self.distance}: {figure_aim} (PSI bin: {bin})", fontsize=36, x=0.45, y=1.1)
                    ax.set_title(f"All Graphs; Hierarchical clustering w/ Ward; {normalization_suffix}", fontsize=20, y=1.04)

                    sanitized_key = bin.replace("<", "less-than").replace(">", "greater-than").replace("=", "equals").replace("&", "and")
                    plt.savefig(f"../outputs/rbp_binding_heatmaps/psi_stratified_heatmaps/{cell_line}_{sanitized_key}_{figure_aim_suffix}_AKA_{normalization_filename_suffix}.png", bbox_inches='tight', dpi=300)
                    plt.show()
                

    def plot_knockdown_responsive_RBP_positional_preferences(self): 

        dpsi_threshold = 0.05
        fdr_threshold = 0.05

        for cell_line in self.cell_lines: 
            kd_responsive_graphs = self.modeling_input_data[cell_line].filter(
                (pl.col("DeltaPSI").abs() > dpsi_threshold) & (pl.col("FDR") < fdr_threshold)
            )
            logger.info(f"{cell_line} - Original row count: {self.modeling_input_data[cell_line].shape[0]}, Filtered row count: {kd_responsive_graphs.shape[0]}")

            for use_unique_exons in [True, False]:
                
                filtered_data = kd_responsive_graphs

                if use_unique_exons:
                    filtered_data = filtered_data.filter(pl.col("RBP_KD_Target") == "CTRL")
                    filtered_data = filtered_data.unique(subset=["Upstream Exon", "Main Exon", "Downstream Exon"])

                    logger.info(f"Using unique 3-exon combinations for CTRLs. New row count: {filtered_data.shape[0]}")           
            
                num_graphs = filtered_data.shape[0]

                for column_normalized in [False, True]:

                    plotting_data = self.convert_features_to_2d_table(
                        self.summate_binding_amount(
                            filtered_data, 
                            total_binding=True
                        ),
                        column_normalized=column_normalized
                    )
                    plotting_data = plotting_data.dropna(axis=1, how='all')
                    assert not plotting_data.isnull().values.any(), "plotting_data contains null values"
                    assert plotting_data.max().max() <= 100, "Some values in the heatmap are greater than 1"

                    plotting_data = self.return_ward_hierarchical_clustering_order(plotting_data)

                    plt.figure(figsize=(35, 5), dpi=300)

                    ax = sns.heatmap(
                        plotting_data,
                        cmap="Blues",
                        mask=plotting_data.isnull(),
                        cbar=True,
                        linewidths=.5,
                        linecolor='black',
                    )
                    ax.set_facecolor('gray')

                    ax.tick_params(axis='x', which='major', labelsize=12, length=6, width=1.5)
                    ax.tick_params(axis='y', which='major', labelsize=18, length=6, width=1.5)

                    cbar = ax.collections[0].colorbar
                    cbar.ax.tick_params(labelsize=16)
                    cbar.ax.set_position([0.76, 0.15, 0.04, 0.7])
                    cbar.ax.set_title("%", fontsize=20)

                    if column_normalized:
                        normalization_suffix = "Column Normalized"
                        figure_aim = "Positional Preference of RBPs"
                        normalization_filename_suffix = "column-normalized"
                        figure_aim_suffix = "positional-preference-of-rbps"
                    else:
                        normalization_suffix = "NOT column normalized"
                        figure_aim = "% Graphs Bound"
                        normalization_filename_suffix = "not-column-normalized"
                        figure_aim_suffix = "percent-graphs-bound"

                    plt.suptitle(f"{cell_line} {self.distance}: 'Responsive KD' {figure_aim}", fontsize=36, x=0.45, y=1.1)
                    
                    exons_used = "Unique 3-Exon Combinations" if use_unique_exons else "All Graphs"
                    ax.set_title(f"{normalization_suffix}; # Graphs: {num_graphs}; {exons_used}; Hierarchical clustering w/ Ward", fontsize=20, y=1.04)

                    plt.savefig(f"../outputs/rbp_binding_heatmaps/knockdown_responsive_heatmaps/{cell_line}_unique_exons_{use_unique_exons}_knockdown_responsive_binding_{normalization_filename_suffix}.png", bbox_inches='tight', dpi=300)
                    plt.show()


    def retrieve_wandb_models(self): 

        if len(glob.glob("../outputs/wandb_results/*"))!=4:
        
            logger.info("Retrieving WandB results...")
    
            run_names = {
                "xgboost": {
                    "K562": 'model-good-tree-16:v0',
                    "HepG2": 'model-likely-shape-14:v0'
                },
                "elasticnet": {
                    "K562": "model-whole-butterfly-17:v0", 
                    "HepG2": 'model-fresh-sunset-13:v0'
                }
            }

            api = wandb.Api()

            for model_type, cell_line_run_names in run_names.items():
                for cell_line, run_name in cell_line_run_names.items():
                    artifact = api.artifact(f"platiglab/rbp-se-castaldi-talk-202502/{run_name}")

                    artifact_dir = pathlib.Path(self.WANDB_DIR) / f"{cell_line}_{model_type}"
                    artifact_dir.mkdir(parents=True, exist_ok=True)

                    artifact.download(root=artifact_dir)

        else: 
            logger.info("FROM CACHE: Loading WandB results ...")

            self.wandb_models = {}

            for model_type in ["elasticnet", "xgboost"]:
                self.wandb_models[model_type] = {}

                for cell_line in self.cell_lines:
                    model_path = pathlib.Path(self.WANDB_DIR) / f"{cell_line}_{model_type}"
                    model_files = list(model_path.glob("*.pkl"))
                    assert len(model_files) == 1, f"Expected exactly one .pkl file, but found {len(model_files)}"

                    self.wandb_models[model_type][cell_line] = pickle.load(open(model_files[0], "rb"))

            logger.success("WandB results loaded successfully.")

    
    def run_elasticnet_with_all_data_and_training_data(self):
        
        if not hasattr(self, "wandb_models"):
            self.retrieve_wandb_models()

        ELASTICNET_COEF_DIR= pathlib.Path("../outputs/elasticnet/coefficients/")
        ELASTICNET_MODEL_DIR = pathlib.Path("../outputs/elasticnet/fitted_models/")
        ELASTICNET_PRED_DIR = pathlib.Path("../outputs/elasticnet/predictions/")

        if len(list(ELASTICNET_COEF_DIR.glob("*.tsv"))) == 2:
            logger.info("FROM CACHE: loading ElasticNet coefficients, fitted model/predictions (trained on just train set)...")

            self.elasticnet_coefficients_matrix = {}
            self.elasticnet_fitted_models = {}
            self.elasticnet_predictions = {}
            
            for cell_line in self.cell_lines:
                file_path = ELASTICNET_COEF_DIR / f"{cell_line}_elasticnet_coefficients.tsv"
                self.elasticnet_coefficients_matrix[cell_line] = pd.read_csv(file_path, sep='\t', index_col=0)

                model_file_path = ELASTICNET_MODEL_DIR / f"{cell_line}_elasticnet_fitted_with_training_data_model.pkl"
                self.elasticnet_fitted_models[cell_line] = pickle.load(open(model_file_path, "rb"))

                pred_file_path = ELASTICNET_PRED_DIR / f"{cell_line}_elasticnet_test_predictions.tsv.gz"
                self.elasticnet_predictions[cell_line] = pd.read_csv(pred_file_path, sep='\t', compression='gzip')  

            logger.success("All ElasticNet info loaded.")

        else: 

            for cell_line in self.cell_lines:
                
                model = self.wandb_models["elasticnet"][cell_line]
                logger.info(f"Running ElasticNet for {cell_line}...")

                params = model.get_params()
                binding_cols = self.binding_cols[cell_line]

                for model_type in ["performance_focused", "coefficient_focused"]: 
                    
                    if model_type == "performance_focused": 
                        X = self.modeling_input_data[cell_line].filter(pl.col("chr").is_in(self.train_set + self.validate_set))
                        
                        X_test = self.modeling_input_data[cell_line].filter(pl.col("chr").is_in(self.test_set)).select(binding_cols + ["index"]).sort("index").to_pandas().set_index("index")
                        y_test = self.modeling_input_data[cell_line].filter(pl.col("chr").is_in(self.test_set)).select("Target_PSI", "index").sort("index").to_pandas().set_index("index")
                        
                        assert (X_test.index == y_test.index).all(), "The index order between X_test and y_test does not match"
                        y_test = y_test["Target_PSI"].astype(float)

                    elif model_type == "coefficient_focused": 
                        X = self.modeling_input_data[cell_line]

                    y = X.select("Target_PSI", "index").sort("index").to_pandas().set_index("index")                    
                    X = X.select(binding_cols + ["index"]).sort('index').to_pandas().set_index("index")

                    assert (X.index == y.index).all(), "The index order between X and y does not match"
                    y = y["Target_PSI"].astype(float)

                    new_model = ElasticNet(**params)
                    new_model.fit(X, y)

                    if model_type == "coefficient_focused":

                        coef_df = pd.DataFrame([new_model.coef_], columns=X.columns)
                        coef_df.index = ["Coefficient"]

                        coef_df = self.convert_features_to_2d_table(coef_df, column_normalized=False)
                        
                        output_path = ELASTICNET_COEF_DIR / f"{cell_line}_elasticnet_coefficients.tsv"
                        coef_df.to_csv(output_path, sep='\t')

                        logger.success(f"ElasticNet coefficients (fit with ALL data) for {cell_line} saved to {output_path}")
                    
                    elif model_type == "performance_focused": 

                        y_pred = new_model.predict(X_test)
                        predictions = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

                        pred_file_path = ELASTICNET_PRED_DIR / f"{cell_line}_elasticnet_test_predictions.tsv.gz"
                        predictions.to_csv(pred_file_path, sep='\t', compression='gzip', index=False)
                        
                        output_path = ELASTICNET_MODEL_DIR / f"{cell_line}_elasticnet_fitted_with_training_data_model.pkl"
                        with open(output_path, "wb") as f:
                            pickle.dump(new_model, f)

                        logger.success(f"ElasticNet predictions/model (fit with TRAINING data) for {cell_line} saved.")
                        

    def plot_elasticnet_coefficients(self): 
        
        if not hasattr(self, "elasticnet_coefficients_matrix"):
            self.run_elasticnet_with_all_data_and_training_data()

        for cell_line, coef_df in self.elasticnet_coefficients_matrix.items():
            plt.figure(figsize=(28, 5), dpi=300)

            assert not coef_df.isnull().values.any(), "There are missing or null values in the coefficient DataFrame"
            
            ax = sns.heatmap(
                coef_df,
                cmap="seismic",
                cbar=True,
                linewidths=.5,
                linecolor='black',
                center=0
            )
            ax.set_facecolor('gray')

            ax.tick_params(axis='x', which='major', labelsize=12, length=6, width=1.5)
            ax.tick_params(axis='y', which='major', labelsize=18, length=6, width=1.5)

            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=16)
            cbar.ax.set_position([0.76, 0.15, 0.04, 0.7])

            plt.suptitle(f"{cell_line} {self.distance}: ElasticNet Coefficients", fontsize=36, x=0.45, y=0.99)
            plt.savefig(f"../outputs/elasticnet/plots/{cell_line}_elasticnet_coefficients.png", bbox_inches='tight', dpi=300)
            plt.show()

            plt.figure(figsize=(7,3), dpi=300)
            
            plt.hist(coef_df.values.flatten(), bins=50, color='gold', edgecolor='black')
            plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5)

            plt.title(f"{cell_line} {self.distance}: Distribution of ElasticNet Coefficients", fontsize=12)
            plt.xlabel("Coefficient Value", fontsize=10)
            plt.ylabel("Frequency", fontsize=10)

            plt.savefig(f"../outputs/elasticnet/plots/{cell_line}_elasticnet_coefficients_histogram.png", bbox_inches='tight', dpi=300)
            plt.show()


    def run_xgboost(self):

        if not hasattr(self, "wandb_models"):
            self.retrieve_wandb_models()
        
        XGBOOST_DIR = pathlib.Path("../outputs/xgboost/model_fit/")

        if len(list(XGBOOST_DIR.glob("*.pkl"))) == 2:
            logger.info("FROM CACHE: loading XGBoost fitted models...")

            self.xgboost_fitted_models = {}
            for cell_line in self.cell_lines:
                file_path = XGBOOST_DIR / f"{cell_line}_xgboost_fitted_model.pkl"
                self.xgboost_fitted_models[cell_line] = pickle.load(open(file_path, "rb"))

            logger.success("XGBoost fitted models loaded successfully.")


        else:

            for cell_line in self.cell_lines:

                brett_model = self.wandb_models["xgboost"][cell_line]

                xgb_params = brett_model.get_params()
                xgb_params['n_jobs'] = self.get_num_cpus()
                
                new_xgb_model = type(brett_model)(**xgb_params)

                logger.info(f"Creating training data for {cell_line}...")
                train_data = self.modeling_input_data[cell_line].filter(pl.col("chr").is_in(self.train_set)).sort("index")
                validate_data = self.modeling_input_data[cell_line].filter(pl.col("chr").is_in(self.validate_set)).sort("index")

                X_train = train_data.select(self.binding_cols[cell_line] + ["index"]).to_pandas().set_index("index")
                y_train = train_data.select("Target_PSI", "index").to_pandas().set_index("index")["Target_PSI"].astype(float)

                X_validate = validate_data.select(self.binding_cols[cell_line] + ["index"]).to_pandas().set_index("index")
                y_validate = validate_data.select("Target_PSI", "index").to_pandas().set_index("index")["Target_PSI"].astype(float)

                assert (X_train.index == y_train.index).all(), "The index order between X_train and y_train does not match"
                assert (X_validate.index == y_validate.index).all(), "The index order between X_validate and y_validate does not match"

                logger.info(f"Fitting XGBoost model for {cell_line} with early stopping...")

                new_xgb_model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_train, y_train), (X_validate, y_validate)],
                    verbose=True
                )

                output_path = XGBOOST_DIR / f"{cell_line}_xgboost_fitted_model.pkl"
                with open(output_path, "wb") as f:
                    pickle.dump(new_xgb_model, f)

                logger.success(f"XGBoost model for {cell_line} saved to {output_path}")


    def calculate_xgboost_performance(self): 
        if not hasattr(self, "xgboost_fitted_models"):
            self.run_xgboost()

        PRED_DIR = pathlib.Path("../outputs/xgboost/test_predictions/")

        if len(list(PRED_DIR.glob("*.tsv.gz"))) == 2:
            logger.info("FROM CACHE: Loading XGBoost predictions...")

            self.xgboost_predictions = {}
            for cell_line in self.cell_lines:
                pred_file_path = PRED_DIR / f"{cell_line}_xgboost_predictions.tsv.gz"

                predictions = pd.read_csv(pred_file_path, sep='\t')
                self.xgboost_predictions[cell_line] = predictions

            logger.success("XGBoost predictions loaded successfully.")

        else:

            self.xgboost_predictions = {}
            for cell_line, xgb_model in self.xgboost_fitted_models.items():

                logger.info(f"Creating test data for {cell_line}...")

                test_data = self.modeling_input_data[cell_line].filter(pl.col("chr").is_in(self.test_set)).sort("index")

                X_test = test_data.select(self.binding_cols[cell_line] + ["index"]).to_pandas().set_index("index")
                y_test = test_data.select("Target_PSI", "index").to_pandas().set_index("index")["Target_PSI"].astype(float)

                assert (X_test.index == y_test.index).all(), "The index order between X_test and y_test does not match"

                logger.info(f"Predicting with XGBoost model for {cell_line}...")

                y_pred = xgb_model.predict(X_test)
                predictions = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

                pred_file_path = PRED_DIR / f"{cell_line}_xgboost_predictions.tsv.gz"
                predictions.to_csv(pred_file_path, sep='\t', compression='gzip', index=False)

            logger.success(f"Predictions for {cell_line} saved to {pred_file_path}")


    def plot_performance(self):

        if not hasattr(self, "xgboost_predictions"):
            self.calculate_xgboost_performance()

        if not hasattr(self, "elasticnet_predictions"):
            self.calculate_elasticnet_performance()

        PERFORMANCE_DIR = "../outputs/performance/"

        for cell_line in self.cell_lines:
            for lognorm in [False, True]:

                fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300, sharex=True, sharey=False)

                models = {
                    "XGBoost": self.xgboost_predictions[cell_line],
                    "ElasticNet": self.elasticnet_predictions[cell_line]
                }

                for ax, (model_name, predictions) in zip(axes, models.items()):

                    if lognorm:
                        norm=LogNorm()
                        stat = 'count'
                    else: 
                        norm=None
                        stat = 'percent'

                    sns.histplot(
                        x=predictions["Actual"],
                        y=predictions["Predicted"],
                        bins=100,
                        cmap="Blues",
                        cbar=True,
                        ax=ax,
                        norm = norm,
                        stat=stat,
                        vmin=None, 
                        vmax=None
                    )

                    ax.plot([predictions["Actual"].min(), predictions["Actual"].max()],
                            [predictions["Actual"].min(), predictions["Actual"].max()],
                            linestyle='--', color='red')
                    r2 = r2_score(predictions["Actual"], predictions["Predicted"])

                    ax.text(
                        0.05, 0.95, f"$R^2$: {r2:.2f}",
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax.transAxes,
                        fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.5)
                    )
                    
                    if model_name == "ElasticNet":
                        ax.axhline(y=0, color='green', linestyle='--')
                        ax.axhline(y=1, color='green', linestyle='--')

                    cbar = ax.collections[0].colorbar
                    if lognorm:
                        cbar.ax.set_title("log10", fontsize=12)
                    else:
                        cbar.ax.set_title("%", fontsize=16)

                    ax.set_title(f"{model_name}", fontsize=18)
                    ax.set_xlabel("Actual", fontsize=18)
                    ax.set_ylabel("Predicted", fontsize=18)

                suffix = "\nNOTE: Data is log10-normed" if lognorm else ""

                plt.suptitle(f"{cell_line}: 'Test Set' Actual vs Predicted{suffix}", fontsize=18)
                plt.tight_layout()
                plt.savefig(f"{PERFORMANCE_DIR}/{cell_line}_log_norm_{lognorm}_model_performance.png", bbox_inches='tight', dpi=300)
                plt.show()

    
    def create_training_strategy_table(self): 

        TABLE_DIR=pathlib.Path("../outputs/training_strategy_tables/")

        fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
        ax.axis('off')

        num_graphs_table_data = []
        for cell_line in self.cell_lines:
            for partition, chr_set in zip(["Train", "Validate", "Test"], [self.train_set, self.validate_set, self.test_set]):
                num_graphs = self.modeling_input_data[cell_line].filter(pl.col("chr").is_in(chr_set)).shape[0]
                num_graphs_table_data.append([cell_line, partition, num_graphs])

        num_graphs_table_data = sorted(num_graphs_table_data, key=lambda x: (x[0], x[1]))

        num_graphs_table = ax.table(cellText=num_graphs_table_data, colLabels=["Cell Line", "Partition", "# Graphs"], cellLoc='center', loc='center')
        num_graphs_table.auto_set_font_size(False)
        num_graphs_table.set_fontsize(12)
        num_graphs_table.scale(1.2, 1.5)

        for key, cell in num_graphs_table.get_celld().items():
            if key[0] == 0:
                cell.set_text_props(weight='bold', fontsize=14)
        
        plt.savefig(TABLE_DIR / "number_graphs_used.png", bbox_inches='tight', dpi=300)
        logger.success(f"Number graphs used table saved as image to {TABLE_DIR / 'number_graphs_used.png'}")

        fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
        ax.axis('off')
        max_len = max(len(self.train_set), len(self.validate_set), len(self.test_set))
        chromosomes_used_data = {
            "Train": sorted(self.train_set) + [""] * (max_len - len(self.train_set)),
            "Validate": sorted(self.validate_set) + [""] * (max_len - len(self.validate_set)),
            "Test": sorted(self.test_set) + [""] * (max_len - len(self.test_set))
        }

        chromosomes_used = ax.table(cellText=[list(row) for row in zip(*chromosomes_used_data.values())], colLabels=list(chromosomes_used_data.keys()), cellLoc='center', loc='center')
        chromosomes_used.auto_set_font_size(False)
        chromosomes_used.set_fontsize(12)
        chromosomes_used.scale(1.2, 1.5)
        
        for key, cell in chromosomes_used.get_celld().items():
            if key[0] == 0:
                cell.set_text_props(weight='bold', fontsize=14)

        plt.savefig(TABLE_DIR / "chromosomes_used.png", bbox_inches='tight', dpi=300)
        logger.success(f"Chromosomes used table saved as image to {TABLE_DIR / 'chromosomes_used.png'}")


    def calculate_stability_of_xgboost_expected_values(self):

        self.shap_expected_values = {}
        pickle_file_path = pathlib.Path("../outputs/shap/stability/shap_expected_values.pkl")

        if pickle_file_path.exists():
            logger.info("FROM CACHE: Loading SHAP expected values...")

            with open(pickle_file_path, "rb") as f:
                self.shap_expected_values = pickle.load(f)
            
            for key, values_dict in self.shap_expected_values.items():
                for cell_line, values in values_dict.items():
                    for i, value in enumerate(values):
                        if not isinstance(value, (int, float)):
                            assert isinstance(value, np.ndarray) and value.size == 1, f"Expected a numpy array of size 1, but got {value}"
                            self.shap_expected_values[key][cell_line][i] = value.item()

            logger.success("SHAP expected values loaded successfully.")

        else:
            logger.info("Calculating SHAP expected values...")
            self.shap_expected_values = {"reinstantiated_training_data": {}, "no_training_data": {}}

            for cell_line in self.cell_lines:
                expected_values = []

                for _ in range(50):
                    X_test = self.modeling_input_data[cell_line].filter(pl.col("chr").is_in(self.test_set)).select(self.binding_cols[cell_line])
                    X_test = X_test.sample(fraction=1, with_replacement=False).to_pandas()
                    
                    explainer = shap.TreeExplainer(self.xgboost_fitted_models[cell_line], data=X_test)
                    expected_values.append(explainer.expected_value)
                
                self.shap_expected_values["reinstantiated_training_data"][cell_line] = expected_values
            
            for cell_line in self.cell_lines:
                expected_values = []

                for _ in range(50):
                    explainer = shap.TreeExplainer(self.xgboost_fitted_models[cell_line])
                    expected_values.append(explainer.expected_value)  
                
                self.shap_expected_values["no_training_data"][cell_line] = expected_values

            with open(pickle_file_path, "wb") as f:
                pickle.dump(self.shap_expected_values, f)

            logger.success("SHAP expected values calculated and saved successfully.")


    def plot_stability_of_xgboost_expected_values(self):

        if not hasattr(self, "shap_expected_values"):
            self.calculate_stability_of_xgboost_expected_values()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharey=True, sharex=True)

        for ax, (key, values_dict) in zip(axes, self.shap_expected_values.items()):
            data = []
            for cell_line, values in values_dict.items():
                for value in values:
                    data.append([cell_line, value])
            
            df = pd.DataFrame(data, columns=["Cell Line", "Expected Value"])

            sns.swarmplot(
                x="Cell Line", 
                y="Expected Value", 
                data=df, 
                ax=ax, 
                color="red", 
                edgecolor="black", 
                size=2, 
                linewidth= 0.5
            )

            
            subplot_title = "Re-retrieving and Randomly Ordering \nTraining Data each Time" if key == "reinstantiated_training_data" else "No Input Data for SHAP Explainer"
            num_points = len(df[df["Cell Line"] == df["Cell Line"].unique()[0]])
            
            ax.set_title(f"{subplot_title} (n={num_points})", fontsize=16)
            ax.set_ylabel('')
            ax.set_xlabel('')

        plt.suptitle("Expected Value of SHAP Explainer by Method", fontsize=20)
        fig.supxlabel("Cell Line", fontsize=16, y=0.02)
        fig.supylabel("Expected Value", fontsize=16, x=-0.02)

        plt.tight_layout()
        plt.show()



    def calculate_xgboost_SHAP_values(self): 

        SHAP_DIR = pathlib.Path("../outputs/shap/local_shap_values/")
        self.shap_values = {}

        if len(list(SHAP_DIR.glob("*.tsv"))) == 2:
            pass
            # logger.info("FROM CACHE: Loading SHAP values...")

            # self.shap_values = {}
            # for cell_line in self.cell_lines:
            #     shap_file_path = SHAP_DIR / f"{cell_line}_shap_values.tsv"
            #     self.shap_values[cell_line] = pd.read_csv(shap_file_path, sep='\t', index_col=0)

            # logger.success("SHAP values loaded successfully.")

        else:


            for cell_line in self.cell_lines:

                logger.info(f"Calculating SHAP values for {cell_line}...")

                X_test = self.modeling_input_data[cell_line].filter(pl.col("chr").is_in(self.train_set)).select(self.binding_cols[cell_line] + ["index"]).sort("index").to_pandas().set_index("index")
                all_data = self.modeling_input_data[cell_line].select(self.binding_cols[cell_line] + ["index"]).sort("index").to_pandas().set_index("index")

                explainer = shap.TreeExplainer(self.xgboost_fitted_models[cell_line], data= X_test, feature_names=X_test.columns)

                shap_values = pl.DataFrame(
                    explainer.shap_values(all_data), 
                    schema=all_data.columns.tolist()
                )
                shap_values = shap_values.with_columns(pl.Series("index", all_data.index).alias("index")).select(["index"] + all_data.columns.tolist())

                output_path = SHAP_DIR / f"{cell_line}_xgboost_shap_values.feather"
                shap_values.write_ipc(output_path, compression='lz4')

                logger.success(f"SHAP values for {cell_line} saved to {output_path}")




    def tmp(self): 
        pass