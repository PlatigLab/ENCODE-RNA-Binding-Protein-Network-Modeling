import polars as pl, glob, matplotlib.pyplot as plt, pandas as pd, re, pathlib, json, matplotlib.colors as mcolors, concurrent.futures, tqdm, os, seaborn as sns, random, argparse, sys, itertools, gc

from dataclasses import dataclass
from loguru import logger
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

@dataclass
class RbpPpiAnalyzer:

    # initate the class with the following parameters
    distance_threshold: int = None

    ##########################################
    # General (non-class specific) variables #
    ##########################################
    FEATHER_CACHE_DIR = "/project/PlatigLab/users/yogi/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/4_Ayans_XGBDT_SHAP_analysis/outputs/__featherv2-cache__/"
    LINEAR_MODEL_DIR = "/project/PlatigLab/data/collaborators/BWH/6_ols_regression_and_xgbdt_models_2024_11/linear-models-ols-2024-11/linear-models-ols-100-0599cbc0/"
    PPI_CACHE_DIR = "../output/ppi_cache_data/"

    cell_lines = ["K562", "HepG2"]

    # RBP PPI 
    rbp_comparisons_file = "/project/PlatigLab/users/yogi/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/4_Ayans_XGBDT_SHAP_analysis/outputs/rbp_comparisons/rbp_comparisons.json"

    # dictionary to rename the splice junction positions to numbers
    splice_junction_position_renaming = {
        "5_left": 1, 
        "5_right": 2, 
        "center_left": 3, 
        "center_right": 4, 
        "3_left": 5, 
        "3_right": 6
    }

    psi_partition_thresholds=[0.1,0.9]

    
    def __post_init__(self):

        self.load_SHAP_data()
        self.load_linear_model_results()

        self.get_total_binding()
        self.check_initial_data_assertions()

        self.load_RBP_PPI_pairs()
        # self.retrieve_rbp_ppi_events_and_controls()


    def load_RBP_PPI_pairs(self):

        with open(self.rbp_comparisons_file, "r") as f:
            rbp_ppi = json.load(f)

        for cell_line in self.cell_lines:
            rbp_ppi[cell_line] = [sorted(pair) for pair in rbp_ppi[cell_line]]
        
        self.rbp_ppi = rbp_ppi

        logger.success(f"FROM CACHE: RBP PPI info loaded")


    def load_SHAP_data(self):
        logger.info(f"FROM CACHE: Loading SHAP data for distance threshold {self.distance_threshold}.")
        
        shap_data = {}

        for cell_line in self.cell_lines: 

            shap_file = f"{self.FEATHER_CACHE_DIR}/{cell_line}-{self.distance_threshold}-shap_data.feather"
            shap_data[cell_line] = pl.read_ipc(shap_file)

        
        self.shap_data = shap_data

        binding_columns = {}
        shap_columns = {}

        for cell_line in self.cell_lines:
            binding_columns[cell_line] = [col for col in self.shap_data[cell_line].columns if col.endswith("_right") or col.endswith("_left")]
            shap_columns[cell_line] = [col for col in self.shap_data[cell_line].columns if col.endswith("_shap")]
        
        self.binding_columns = binding_columns
        self.shap_columns = shap_columns

        logger.success(f"Loaded SHAP data for distance threshold: {self.distance_threshold}")
    

    def load_linear_model_results(self): 
        logger.info(f"FROM CACHE: Loading linear model predictions for distance threshold {self.distance_threshold}.")

        linear_coefficients = {}
        for cell_line in self.cell_lines: 

            linear_coefficients_file = glob.glob(f"{self.LINEAR_MODEL_DIR}/{cell_line}-{self.distance_threshold}-*-linear-model-beta.dat")
            assert len(linear_coefficients_file) == 1

            linear_coefficients[cell_line] = pd.read_csv(linear_coefficients_file[0], sep=",", index_col=0)
            linear_coefficients[cell_line].index.name = "Feature"
        
        self.linear_coefficients = linear_coefficients

        linear_model_results = {}
        for cell_line in self.cell_lines:
            linear_model_results_file = f"{self.FEATHER_CACHE_DIR}/{cell_line}-{self.distance_threshold}-linear_model_results.feather"
            linear_model_results[cell_line] = pl.read_ipc(linear_model_results_file)

        self.linear_model_results = linear_model_results

        logger.success(f"Retrieved linear model predictions for distance threshold: {self.distance_threshold}.")

    
    def get_total_binding(self):

        logger.info("Calculating total binding for SHAP data and linear model results.")

        if not hasattr(self, 'shap_data'):
            self.load_SHAP_data()

        if not hasattr(self, 'linear_model_results'):
            self.load_linear_model_results()

        for cell_line in self.cell_lines:
                
            shap_data = self.shap_data[cell_line]
            linear_model_results = self.linear_model_results[cell_line]

            shap_data = shap_data.with_columns(
                pl.sum_horizontal(pl.col(self.binding_columns[cell_line])).alias("Total Binding")
            )

            linear_model_results = linear_model_results.with_columns(
                pl.sum_horizontal(pl.col(self.binding_columns[cell_line])).alias("Total Binding")
            )

            self.shap_data[cell_line] = shap_data
            self.linear_model_results[cell_line] = linear_model_results

        logger.success("Finished calculating total binding for SHAP data and linear model results.")


    def check_initial_data_assertions(self): 

        logger.info("Checking initial data assertions.")
        
        for cell_line in self.cell_lines:
            
            # Check if SHAP data and linear model results have the same number of rows
            assert self.shap_data[cell_line].shape[0] == self.linear_model_results[cell_line].shape[0], logger.error(f"SHAP data and linear model results do not have the same number of rows for {cell_line}.")

            shap_binding_columns = [col for col in self.shap_data[cell_line].columns if col.endswith("_right") or col.endswith("_left")]
            linear_model_binding_columns = [col for col in self.linear_model_results[cell_line].columns if col.endswith("_right") or col.endswith("_left")]

            # Check if SHAP data and linear model results have the same binding columns
            assert set(shap_binding_columns) == set(linear_model_binding_columns), logger.error(f"SHAP data and linear model results do not have the same binding columns for {cell_line}.")

            # Check if SHAP data and linear model results have only unique graph indices
            assert self.shap_data[cell_line]["graph_index"].n_unique() == self.shap_data[cell_line].shape[0], logger.error(f"Duplicate values found in SHAP data 'graph_index' for {cell_line}.")
            assert self.linear_model_results[cell_line]["graph_index"].n_unique() == self.linear_model_results[cell_line].shape[0], logger.error(f"Duplicate values found in linear model results 'graph_index' for {cell_line}.")

            # Check if SHAP data and linear model results have the same graph indices
            assert set(self.shap_data[cell_line]["graph_index"]) == set(self.linear_model_results[cell_line]["graph_index"]), logger.error(f"'graph_index' values do not match between SHAP data and linear model results for {cell_line}.")

            # take the columns "graph_index" and "Total Binding" from both datasets and convert this to dictionary and assert that they are the same for both datasets
            shap_total_binding = dict(zip(self.shap_data[cell_line]["graph_index"], self.shap_data[cell_line]["Total Binding"]))
            linear_model_total_binding = dict(zip(self.linear_model_results[cell_line]["graph_index"], self.linear_model_results[cell_line]["Total Binding"]))

            assert shap_total_binding == linear_model_total_binding, logger.error(f"Total binding values do not match between SHAP data and linear model results for {cell_line}.")

            # check that there are no null or missing values for the columns "target" and "psi_hat"
            for column in ["target", "psi_hat"]:
                assert self.shap_data[cell_line][column].has_nulls() == False, logger.error(f"Null values found in '{column}' column for SHAP data for {cell_line}.")
                assert self.linear_model_results[cell_line][column].has_nulls() == False, logger.error(f"Null values found in '{column}' column for linear model results for {cell_line}.")

            # take all rows where total binding is 0 and check that the only value for has_RBP_KD is True
            zero_binding_rows_shap = self.shap_data[cell_line].filter(pl.col("Total Binding") == 0)
            assert zero_binding_rows_shap["has_RBP_KD"].unique().to_list() == [True], logger.error(f"Found non-True values in 'has_RBP_KD' for rows with 0 total binding in SHAP data for {cell_line}.")
            assert all(zero_binding_rows_shap["RBP_KD"] != "NONE"), logger.error(f"Found non-NONE values in 'RBP_KD' for rows with 0 total binding in SHAP data for {cell_line}.")

            zero_binding_rows_linear = self.linear_model_results[cell_line].filter(pl.col("Total Binding") == 0)
            assert zero_binding_rows_linear["has_RBP_KD"].unique().to_list() == [True], logger.error(f"Found non-True values in 'has_RBP_KD' for rows with 0 total binding in linear model results for {cell_line}.")
            assert all(zero_binding_rows_linear["RBP_KD"] != "NONE"), logger.error(f"Found non-NONE values in 'RBP_KD' for rows with 0 total binding in linear model results for {cell_line}.")
            
        logger.success("Initial data assertions passed.")


    def check_binding_graphs_equal_for_shap_vs_linear_regression(self): 

        logger.info("Checking if binding graphs are equal for SHAP vs Linear Regression.")

        for cell_line in self.cell_lines:
            tmp_shap = self.shap_data[cell_line].sort("graph_index").select(self.binding_columns[cell_line])
            tmp_linear = self.linear_model_results[cell_line].sort("graph_index").select(self.binding_columns[cell_line])

            if tmp_shap.equals(tmp_linear): 
                logger.info(f"Binding graphs match for {cell_line}.")
                
            else: 
                logger.error(f"Binding graphs do not match for {cell_line}.")
            

    def check_mismatch_graphs(self):  

        MISMATCH_CACHE_FILE= "../output/mismatch_graph_stats/mismatch_stats.tsv"

        if pathlib.Path(MISMATCH_CACHE_FILE).exists():

            logger.info("FROM CACHE: Loading mismatch graph stats.")
            mismatched_graphs = pd.read_csv(MISMATCH_CACHE_FILE, sep="\t")

            self.mismatched_graphs = mismatched_graphs
        
        else: 

            if not hasattr(self, 'shap_data'):
                self.load_SHAP_data()

            logger.info("Calculating number of mismatch graphs for each 3-exon combination.")
            mismatched_graphs = {}

            for cell_line in self.cell_lines:

                subset_data = self.shap_data[cell_line].filter(pl.col("RBP_KD") == "NONE")
                
                unique_combinations = subset_data.select(["ENSE", "ENSE_UP", "ENSE_DN"]).unique().to_dict(as_series=False)
                unique_combinations = list(zip(unique_combinations["ENSE"], unique_combinations["ENSE_UP"], unique_combinations["ENSE_DN"]))
                
                def process_combination(combination, cell_line, data):
                    ense, ense_up, ense_dn = combination

                    combination_subset = data.filter(
                        (pl.col("ENSE") == ense) & 
                        (pl.col("ENSE_UP") == ense_up) & 
                        (pl.col("ENSE_DN") == ense_dn)
                    )

                    return [combination, combination_subset.select(self.binding_columns[cell_line]).unique().shape[0]]

                with concurrent.futures.ThreadPoolExecutor(max_workers=self.get_number_SLURM_CPUs()) as executor:
                    mismatched_graphs[cell_line] = list(tqdm.tqdm(executor.map(lambda comb: process_combination(comb, cell_line, subset_data), unique_combinations), total=len(unique_combinations), desc=f"Processing shap_data for {cell_line}"))

                mismatched_graphs[cell_line] = pd.DataFrame(mismatched_graphs[cell_line], columns=["Combination", "Mismatched Graphs"])
            
            mismatched_graphs = pd.concat(mismatched_graphs, keys=self.cell_lines, names=["Cell Line"]).reset_index(level=0)
            mismatched_graphs.to_csv(MISMATCH_CACHE_FILE, sep="\t", index=False)

    
    def plot_mismatch_graphs(self):

        if not hasattr(self, 'mismatched_graphs'):
            self.check_mismatch_graphs()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=True, dpi=200)

        combined_data = []

        for ax, cell_line in zip(axes, self.cell_lines):
            subset_data = self.mismatched_graphs[self.mismatched_graphs["Cell Line"] == cell_line]
            subset_data.loc[:, "Mismatched Graphs"] = subset_data["Mismatched Graphs"].astype(int)
            
            value_counts = subset_data["Mismatched Graphs"].value_counts().sort_index()
            value_counts_percentage = (value_counts / value_counts.sum()) * 100

            ax.bar(value_counts.index.astype(str), value_counts_percentage.values, width=0.5)
            ax.set_title(f"{cell_line}", fontsize=16)

            num_mismatched = subset_data[subset_data["Mismatched Graphs"] > 1].shape[0]
            total_rows = subset_data.shape[0]
            percentage_mismatched = (num_mismatched / total_rows) * 100

            ax.text(0.5, 0.95, f"> 1 Unique Graph: {num_mismatched}\n# Total: {total_rows}\nPercent Wrong: {percentage_mismatched:.2f}%", 
                transform=ax.transAxes, verticalalignment='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

            for value, count in value_counts.items():
                combined_data.append([cell_line, value, count, value_counts_percentage[value]])

        fig.supxlabel("# Unique Graphs per SE Combination", fontsize=16)
        fig.supylabel("% of Data", fontsize=16)
        plt.suptitle("# Unique Graphs per SE Combination", y=1.0, fontsize=20)

        plt.tight_layout()
        plt.show()

        mismatch_df = pd.DataFrame(combined_data, columns=["Cell Line", "Mismatched Graphs", "Count", "Percentage"])
        return mismatch_df


    def get_number_SLURM_CPUs(self):
        slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")

        if slurm_cpus is not None:
            return int(slurm_cpus)
        else:
            logger.warning("SLURM_CPUS_PER_TASK environment variable not set. Defaulting to 1 CPU.")
            return 1


    def delete_non_PPI_data(self): 

        if hasattr(self, 'shap_data'):
            del self.shap_data
        
        if hasattr(self, 'linear_model_results'):
            del self.linear_model_results
        
        if hasattr(self, 'binding_columns'):
            del self.binding_columns
        
        if hasattr(self, 'shap_columns'):
            del self.shap_columns
        
        gc.collect()



    def _cache_to_featherv2(self, df, file_path):

        if not pathlib.Path(file_path).exists():
            logger.info(f"Caching to Feather V2 (Arrow): {file_path}")

            df.write_ipc(file_path, compression="lz4")
            logger.success(f"Finished creating Feather V2 (Arrow) file for {file_path}")

        else:
            logger.warning(f"Feather V2 (Arrow) file already exists: {file_path}")


    def retrieve_rbp_cobinding(self, data, rbp_ppi_pair, position1, position2):
        assert position1==position2, logger.error(f"Position 1 and Position 2 must be the same.")

        rbp1, rbp2 = sorted(rbp_ppi_pair)
        rbp1 = rbp1.upper()
        rbp2 = rbp2.upper()

        search_columns = [f"{rbp1}_{position1}", f"{rbp2}_{position2}"]

        assert all(col in data.columns for col in search_columns), logger.error(f"Columns {search_columns} not found in DataFrame")

        tmp_data = data.filter(
            (pl.col(search_columns[0]) == 1) & (pl.col(search_columns[1]) == 1)
        ).select("graph_index")

        tmp_data = tmp_data.with_columns(
            [
                pl.lit(f"{rbp1}-{rbp2}").alias("RBP Pair"),
                pl.lit(f"{self.splice_junction_position_renaming[position1]}").alias("Position"), 
                pl.lit("Same Pos. PPI").alias("PPI Analysis Category")
            ]
        ) 

        return tmp_data
    

    def retrieve_single_binders(self, data, rbp_ppi_pair, position):

        rbp1, rbp2 = sorted(rbp_ppi_pair)
        rbp1 = rbp1.upper()
        rbp2 = rbp2.upper()

        search_columns = [f"{rbp1}_{position}", f"{rbp2}_{position}"]
        assert all(col in data.columns for col in search_columns), logger.error(f"Columns {search_columns} not found in DataFrame")

        rbp1_single_binders = data.filter(
            (pl.col(search_columns[0]) == 1) & (pl.col(search_columns[1]) == 0)
        ).select("graph_index")

        rbp1_single_binders = rbp1_single_binders.with_columns(
            [
                pl.lit(f"{rbp1}-{rbp2}").alias("RBP Pair"),
                pl.lit(f"{self.splice_junction_position_renaming[position]}").alias("Position"), 
                pl.lit(f"{rbp1} Only").alias("PPI Analysis Category")
            ]
        )

        rbp2_single_binders = data.filter(
            (pl.col(search_columns[0]) == 0) & (pl.col(search_columns[1]) == 1)
        ).select("graph_index")

        rbp2_single_binders = rbp2_single_binders.with_columns(
            [
                pl.lit(f"{rbp1}-{rbp2}").alias("RBP Pair"),
                pl.lit(f"{self.splice_junction_position_renaming[position]}").alias("Position"),
                pl.lit(f"{rbp2} Only").alias("PPI Analysis Category")
            ]
        )

        return pl.concat([rbp1_single_binders, rbp2_single_binders], how="vertical")


    def retrieve_rbp_ppi_events_and_controls(self): 

        if len(glob.glob(f"{self.PPI_CACHE_DIR}/*-{self.distance_threshold}-*")) == 4:
            
            self.delete_non_PPI_data()

            logger.info("FROM CACHE: Loading RBP PPI events and controls.")

            linear_ppi = {}
            xgboost_ppi = {}

            binding_columns = {}
            shap_columns = {}

            for cell_line in self.cell_lines: 
                linear_ppi[cell_line] = pl.read_ipc(f"{self.PPI_CACHE_DIR}/{cell_line}-{self.distance_threshold}-linear-ppi_events_and_controls.feather")
                xgboost_ppi[cell_line] = pl.read_ipc(f"{self.PPI_CACHE_DIR}/{cell_line}-{self.distance_threshold}-xgboost-ppi_events_and_controls.feather")

                binding_columns[cell_line] = [col for col in xgboost_ppi[cell_line].columns if col.endswith("_right") or col.endswith("_left")]
                shap_columns[cell_line] = [col for col in xgboost_ppi[cell_line].columns if col.endswith("_shap")]

            self.linear_ppi = linear_ppi
            self.xgboost_ppi = xgboost_ppi

            self.binding_columns = binding_columns
            self.shap_columns = shap_columns

            for cell_line in self.cell_lines:
                assert self.linear_ppi[cell_line].shape[0] == self.xgboost_ppi[cell_line].shape[0], logger.error(f"Linear PPI and XGBoost PPI do not have the same number of rows for {cell_line}.")
                assert set(self.linear_ppi[cell_line]["graph_index"]) == set(self.xgboost_ppi[cell_line]["graph_index"]), logger.error(f"'graph_index' values do not match between Linear PPI and XGBoost PPI for {cell_line}.")
                assert self.linear_ppi[cell_line].select(["graph_index", "RBP Pair", "Position", "PPI Analysis Category"]).is_duplicated().any() == False, logger.error(f"Duplicated values found in Linear PPI for {cell_line}.")
                assert self.xgboost_ppi[cell_line].select(["graph_index", "RBP Pair", "Position", "PPI Analysis Category"]).is_duplicated().any() == False, logger.error(f"Duplicated values found in XGBoost PPI for {cell_line}.")
                assert self.linear_ppi[cell_line].select(["graph_index", "RBP Pair", "Position", "PPI Analysis Category"]).equals(self.xgboost_ppi[cell_line].select(["graph_index", "RBP Pair", "Position", "PPI Analysis Category"])), logger.error(f"Linear PPI and XGBoost PPI do not have the same values for {cell_line}.")

                logger.info(f"{cell_line}\nLinear PPI shape: {self.linear_ppi[cell_line].shape} | XGBoost PPI shape: {self.xgboost_ppi[cell_line].shape}")

        else: 

            logger.info("NO CACHE... Hence, retrieving RBP PPI events and controls for both cell lines.")

            if not hasattr(self, 'linear_model_results'):
                self.load_linear_model_results()

            if not hasattr(self, 'shap_data'):
                self.load_SHAP_data()

            for cell_line in self.cell_lines: 

                logger.info(f"Retrieving RBP PPI events and controls for {cell_line}.")

                input_data = self.shap_data[cell_line].select(self.binding_columns[cell_line] + ["graph_index"])

                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.get_number_SLURM_CPUs()) as executor:
                    same_pos_ppi = {
                        executor.submit(self.retrieve_rbp_cobinding, input_data, rbp_ppi_pair, position_key, position_key): (rbp_ppi_pair, position_key) for rbp_ppi_pair in self.rbp_ppi[cell_line] for position_key in self.splice_junction_position_renaming.keys()
                    }

                    for future in tqdm.tqdm(concurrent.futures.as_completed(same_pos_ppi), total=len(same_pos_ppi), desc="Same Position PPI Events"):
                        results.append(future.result())
                
                same_pos_ppi = pl.concat(results, how="vertical")
                assert same_pos_ppi.is_duplicated().any() == False
                
                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.get_number_SLURM_CPUs()) as executor:
                    futures = []
                    for rbp1, rbp2 in self.rbp_ppi[cell_line]:
                        for position in self.splice_junction_position_renaming.keys():
                            futures.append(executor.submit(self.retrieve_single_binders, input_data, (rbp1, rbp2), position))

                    for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Single Binder Events"):
                        results.append(future.result())

                single_binders = pl.concat(results, how="vertical")
                assert single_binders.is_duplicated().any() == False

                final_data = pl.concat(
                    [same_pos_ppi, single_binders], 
                    how="diagonal"
                ).sort("graph_index")
                assert final_data.is_duplicated().any() == False

                self._cache_to_featherv2(final_data, f"{self.PPI_CACHE_DIR}/PPI-ASSIGNMENTS-{cell_line}-{self.distance_threshold}.feather")

                for model_type, df in [("linear", self.linear_model_results[cell_line]), ("xgboost", self.shap_data[cell_line])]:

                    df = df.join(final_data, on="graph_index", how="left", validate="1:m")
                    df = df.sort("graph_index")

                    self._cache_to_featherv2(df, f"{self.PPI_CACHE_DIR}/{cell_line}-{self.distance_threshold}-{model_type}-ppi_events_and_controls.feather")
        
        logger.success("Finished retrieving RBP PPI events and controls.")

    
    def amount_binding_vs_PSI(self): 

        for title_prefix in ["KD + CTRL:", "CTRL ONLY:"]:

            if title_prefix == "KD + CTRL:":
                iter_data = self.shap_data
            elif title_prefix == "CTRL ONLY:":
                iter_data = {cell_line: self.shap_data[cell_line].filter(pl.col("RBP_KD") == "NONE") for cell_line in self.cell_lines}

            logger.info(f"Plotting amount of binding vs PSI for {title_prefix.strip(':')}")

            fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=False, sharey=True, dpi=200)

            for ax, cell_line in zip(axes, self.cell_lines):
                data = iter_data[cell_line].select(["Total Binding", "target"]).to_pandas()
                hb = ax.hexbin(data["Total Binding"], data["target"], gridsize=50, cmap='viridis', mincnt=1, norm = mcolors.LogNorm())
                
                ax.set_title(f"{cell_line}", fontsize=20, pad=20)

                cb = fig.colorbar(hb, ax=ax)
                cb.ax.set_title('Bin Counts', fontsize=12,)

                num_points = len(data)
                spearman_corr, _ = spearmanr(data["Total Binding"], data["target"])

                ax.text(0.95, 0.6, f"Spearman r: {spearman_corr:.2f}\n# Points: {num_points}", 
                        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

            fig.supxlabel("# Bindings per Graph", fontsize=20)
            fig.supylabel("PSI", fontsize=20, x=0.01)
            plt.suptitle(f"{title_prefix} Amount of Binding vs. PSI", fontsize=24, y=1.0)

            plt.tight_layout()
            plt.show()

            
            fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharex=True, sharey=True, dpi=200)

            for ax, cell_line in zip(axes, self.cell_lines):
                data = iter_data[cell_line].select(["Total Binding", "target"]).filter(pl.col("Total Binding") < 5).to_pandas()
                hb = ax.hexbin(data["Total Binding"], data["target"], gridsize=40, cmap='viridis', mincnt=1, norm=mcolors.LogNorm())
                
                ax.set_title(f"{cell_line}", fontsize=14, pad=20)

                cb = fig.colorbar(hb, ax=ax)
                cb.ax.set_title('Bin Counts', fontsize=8,)

                num_points = len(data)
                spearman_corr, _ = spearmanr(data["Total Binding"], data["target"])

                ax.text(0.95, 0.6, f"Spearman r: {spearman_corr:.2f}\n# Points: {num_points}", 
                        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

            fig.supxlabel("# Bindings per Graph", fontsize=10)
            fig.supylabel("PSI", fontsize=10, x=0.01)
            plt.suptitle(f"{title_prefix} Amount of Binding vs. PSI", fontsize=16, y=1.0)

            plt.tight_layout()
            plt.show()

            logger.info("Creating violinplot plot for PSI per each binding amount category.")

            number_of_bins = 15
            combined_data = []

            for cell_line in self.cell_lines:
                data = iter_data[cell_line].select(["Total Binding", "target"]).with_columns(
                    pl.when(pl.col("Total Binding") < number_of_bins)
                        .then(pl.col("Total Binding"))
                        .otherwise(number_of_bins)
                    .alias("Binding Amount Categories")
                )
                
                data = data.with_columns(pl.lit(cell_line).alias("Cell Line"))

                combined_data.append(data)

            combined_data = pl.concat(combined_data).sort(["Cell Line", "Binding Amount Categories"]).with_columns(
                pl.col("Binding Amount Categories").cast(pl.Utf8)
            ).to_pandas()

            combined_data["Binding Amount Categories"] = combined_data["Binding Amount Categories"].replace(str(number_of_bins), f"> {number_of_bins}")

            fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=200, sharey=True, sharex=True)

            for ax, cell_line in zip(axes, self.cell_lines):
                data = combined_data[combined_data["Cell Line"] == cell_line]

                sns.violinplot(x="Binding Amount Categories", y="target", data=data, palette=["lightgreen"], ax=ax, inner=None)
                sns.boxplot(x="Binding Amount Categories", y="target", data=data, color="gray", ax=ax, width=0.1, showcaps=False, boxprops={'facecolor':'gray'}, flierprops={'marker': 'o', 'markersize': 2})

                ax.set_title(f"{cell_line}", fontsize=20)
                ax.set_ylim(-0.2, 1.4)

                ax.set_xlabel("")
                ax.set_ylabel("")

                total_count = data.shape[0]
                for category in data["Binding Amount Categories"].unique():
                    category_count = data[data["Binding Amount Categories"] == category].shape[0]
                    percentage = (category_count / total_count) * 100
                    high_target_percentage = ((data[(data["Binding Amount Categories"] == category) & (data["target"] > 0.9)].shape[0]) / category_count) * 100

                    ax.text(category, 0.92, f"{percentage:.1f}%", ha='center', va='bottom', color="blue", fontsize=10, transform=ax.get_xaxis_transform())
                    ax.text(category, 0.87, f"{high_target_percentage:.1f}%", ha='center', va='bottom', color="red", fontsize=10, transform=ax.get_xaxis_transform())

                ax.axhline(0, color='black', linestyle='--', linewidth=2)
                ax.axhline(1, color='black', linestyle='--', linewidth=2)

            plt.suptitle(f"{title_prefix} Distributions of PSI per # Bindings", fontsize=24, y=1.0)
            fig.supxlabel("# Bindings per Event", fontsize=20)
            fig.supylabel("PSI", fontsize=20, x=0.01)

            plt.tight_layout()
            plt.show()

            logger.info("Plotting cell-line-comparison-specific version of violinplot for distribution of PSI per binding amount category")

            plt.figure(figsize=(20, 6), dpi=200)
            sns.violinplot(x="Binding Amount Categories", y="target", hue="Cell Line", data=combined_data, palette=["#0072B2", "#D55E00"], split=True, gap=0.1)

            plt.axhline(0, color='red', linestyle='--', linewidth=2)
            plt.axhline(1, color='red', linestyle='--', linewidth=2)

            plt.xlabel("# Bindings per Event", fontsize = 24, labelpad=10)
            plt.ylabel("PSI", fontsize=24, labelpad=10)
            plt.title(f"{title_prefix} Distributions of PSI per # Bindings", fontsize=30, pad=20)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)

            plt.tight_layout()
            plt.show()

            logger.success(f"Finished plotting amount of binding vs PSI for {title_prefix.strip(':')}")
            

    def calculate_basic_ppi_stats(self):
        logger.info("Calculating basic PPI stats.")

        if not hasattr(self, 'linear_ppi') or not hasattr(self, 'xgboost_ppi'):
            self.retrieve_rbp_ppi_events_and_controls()


        for y_lim in [False, True]: 

            fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, sharex=True, dpi=200)

            for ax, cell_line in zip(axes, self.cell_lines):
                data = self.xgboost_ppi[cell_line].select(["RBP Pair", "PPI Analysis Category"]).filter(pl.col("PPI Analysis Category") != "No Assignment")

                counts = data.group_by(["RBP Pair", "PPI Analysis Category"]).agg(pl.count()).to_pandas()
                counts = counts.rename(columns={"count": "Count"})

                for idx, row in counts.iterrows():
                    if "PPI" not in row["PPI Analysis Category"]:
                        rbp_pair = row["RBP Pair"].split("-")

                        assert row["PPI Analysis Category"] in rbp_pair

                        if row["PPI Analysis Category"] == rbp_pair[0]:
                            counts.at[idx, "PPI Analysis Category"] = "1st Only"
                        elif row["PPI Analysis Category"] == rbp_pair[1]:
                            counts.at[idx, "PPI Analysis Category"] = "2nd Only"

                if y_lim: 
                    counts = counts[counts["Count"] < 100000]

                counts = counts.sort_values(by=["PPI Analysis Category", "RBP Pair",])

                category_order = ["Same Pos. PPI", "Diff. Pos. PPI", "1st Only", "2nd Only"]

                sns.stripplot(x="PPI Analysis Category", y="Count", data=counts, ax=ax, order=category_order, edgecolor="black", alpha=0.4, jitter=True, size=4)
                
                ax.set_title(f"{cell_line} (# Possible PPIs: {len(self.rbp_ppi[cell_line])})", fontsize=16)
                ax.set_ylabel("")
                ax.set_xlabel("")

                for category in category_order:
                    unique_rb_pairs = counts[counts["PPI Analysis Category"] == category]["RBP Pair"].nunique()

                    if y_lim: 
                        y_coord = 90000
                    else: 
                        y_coord = ax.get_ylim()[1] * 0.90


                    ax.text(category_order.index(category)-0.05, y_coord, f"{unique_rb_pairs}", 
                            ha='center', va='top', fontsize=14, color='red')


                print(counts.sort_values("Count", ascending=False).head(n=10))

            if y_lim:
                notice_str = "NOTE: y-axis is limited to 100,000"
            else: 
                notice_str = ""

            plt.suptitle(f"Counts per Category + RBP-Pair Combination\n{notice_str}", fontsize=20)
            fig.supxlabel("PPI Analysis Category", fontsize=16)
            fig.supylabel("# Category Counts per RBP-Pair", fontsize=12, x=-0.01)


            plt.tight_layout()
            plt.show()

        


    def tmp(self): 
        pass


if __name__ == "__main__":

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(sys.stderr, level="ERROR")

    parser = argparse.ArgumentParser(description="RBP PPI Analyzer")
    parser.add_argument('--distance', type=int, required=True, help='Distance threshold for analysis')
    parser.add_argument('--parallel-task', type=str, required=False, help='Parallel task to run')
    args = parser.parse_args()

    analyzer = RbpPpiAnalyzer(distance_threshold=args.distance)

    match args.parallel_task:

        case "mismatch_graphs":
            analyzer.check_mismatch_graphs()

        case "compare_model_binding_graphs": 
            analyzer.check_binding_graphs_equal_for_shap_vs_linear_regression()

