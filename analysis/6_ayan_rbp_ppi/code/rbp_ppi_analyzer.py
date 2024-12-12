import glob, re, pathlib, json, concurrent.futures, tqdm, os, random, argparse, sys, itertools, gc, scipy
import polars as pl, matplotlib.pyplot as plt, pandas as pd, matplotlib.colors as mcolors, numpy as np, statsmodels.api as sm, seaborn as sns

from dataclasses import dataclass
from loguru import logger
from sklearn.metrics import r2_score

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

    position_inverted_dict = {str(v): k for k, v in splice_junction_position_renaming.items()}

    psi_partition_thresholds=[0.1,0.9]

    
    def __post_init__(self):

        # self.load_SHAP_data()
        # self.load_linear_model_results()

        # self.get_total_binding()
        # self.check_initial_data_assertions()

        self.load_RBP_PPI_pairs()
        self.retrieve_rbp_ppi_events_and_controls(test_only=True)


    def load_RBP_PPI_pairs(self):

        with open(self.rbp_comparisons_file, "r") as f:
            rbp_ppi = json.load(f)

        for cell_line in self.cell_lines:
            rbp_ppi[cell_line] = [tuple(sorted(pair)) for pair in rbp_ppi[cell_line]]
        
        self.rbp_ppi = rbp_ppi

        logger.success(f"FROM CACHE: RBP PPI info loaded\n RBP PPIs in K562: {len(self.rbp_ppi['K562'])}\nRBP PPIs in HepG2: {len(self.rbp_ppi['HepG2'])}")


    def check_non_tested_RBPs(self):
        
        uniprot_mapping = pd.read_csv("../../../inputs/RBP-RBP_PPI/uniprot_mapping.tsv", sep="\t")
        uniprot_mapping["Gene name"] = uniprot_mapping["Gene name"].str.lower()
        uniprot_mapping["Gene Synonym"] = uniprot_mapping["Gene Synonym"].str.lower()

        all_rbps_screened = pd.read_excel("../../../inputs/RBP-RBP_PPI/all_rbps_screened.xlsx")
        all_rbps_screened = set(all_rbps_screened["Gene Symbol"].str.lower())

        for cell_line in self.cell_lines:

            rbps = set()
            rbps.update(
                [col.split('_')[0].lower() for col in self.binding_columns[cell_line]]
            )

            missing_rbps = []
            for rbp in rbps: 
                if rbp not in all_rbps_screened:
                    tmp_df = uniprot_mapping[(uniprot_mapping["Gene name"]==rbp) | (uniprot_mapping["Gene Synonym"]==rbp)]
                    assert len(tmp_df) > 0, print(rbp)

                    tmp_all_synonyms = set([name for name in (tmp_df["Gene name"].to_list() + tmp_df["Gene Synonym"].to_list()) if name!=rbp])
                    if not any(synonym in all_rbps_screened for synonym in tmp_all_synonyms):
                        missing_rbps.append(rbp)
            
            logger.info(f"{(len(missing_rbps) / len(rbps))*100:.2f}% of RBPs in {cell_line} were not tested. RBPs: {missing_rbps}")


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

    
    def delete_PPI_data(self): 
        if hasattr(self, 'xgboost_ppi'):
            del self.xgboost_ppi

        if hasattr(self, 'linear_ppi'):
            del self.linear_ppi

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


    def retrieve_rbp_ppi_events_and_controls(self, test_only=None): 

        # minimum number of same position PPI examples needed 
        # to consider the pair-position combination 
        MIN_EXAMPLES_THRESHOLD=20

        assert test_only in [True, False], logger.error("test_only parameter must be set to True or False.")

        PPI_MISSING_SUMMARY_DIR = "../output/ppi_no_examples_summary/"

        if len(glob.glob(f"{self.PPI_CACHE_DIR}/*-{self.distance_threshold}-*")) == 4:
            
            self.delete_non_PPI_data()

            if test_only:
                data_label = "---TEST----"
            elif not test_only:
                data_label = "----ALL----"

            logger.info(f"FROM CACHE: Loading {data_label} RBP PPI events and controls.")

            linear_ppi = {}
            xgboost_ppi = {}

            binding_columns = {}
            shap_columns = {}

            def parallel_read_ppi_files(cell_line, distance_threshold, model_type, test_only):
                tmp_df = pl.scan_ipc(f"{self.PPI_CACHE_DIR}/{cell_line}-{distance_threshold}-{model_type}-ppi_events_and_controls.feather")

                if not test_only:
                    return tmp_df.collect(streaming=True)

                elif test_only:
                    tmp_df = tmp_df.filter(pl.col("Data Partition") == "test").collect(streaming=True)
            
                    result = (
                        tmp_df.filter(pl.col("PPI Analysis Category") == "Same Pos. PPI")
                        .group_by(["RBP Pair", "Position"])
                        .agg(pl.count())
                        .filter(pl.col("count") >= MIN_EXAMPLES_THRESHOLD)
                        .select(["RBP Pair", "Position"])
                    ).unique()

                    filtered_data = tmp_df.join(result, on=["RBP Pair", "Position"], how="inner")
                    filtered_data = pl.concat([filtered_data, tmp_df.filter(pl.col("RBP Pair").is_null())], how="vertical")
                    assert all(filtered_data["Data Partition"] == "test"), logger.error(f"Not all values in 'Data Partition' column are 'test' for {cell_line}.")

                    return filtered_data

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.get_number_SLURM_CPUs()) as executor:
                linear_futures = {
                    executor.submit(parallel_read_ppi_files, cell_line, self.distance_threshold, "linear", test_only): cell_line for cell_line in self.cell_lines
                }
                xgboost_futures = {
                    executor.submit(parallel_read_ppi_files, cell_line, self.distance_threshold, "xgboost", test_only): cell_line for cell_line in self.cell_lines
                }

                for future in concurrent.futures.as_completed({**linear_futures, **xgboost_futures}):
                    cell_line = linear_futures.get(future) or xgboost_futures.get(future)
                    df = future.result()

                    if future in linear_futures:
                        linear_ppi[cell_line] = df
                    else:
                        xgboost_ppi[cell_line] = df
            
            for cell_line in self.cell_lines:
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

            with open(f"{PPI_MISSING_SUMMARY_DIR}/missing_data_stats_{self.distance_threshold}.json", "r") as f:
                missing_stats_dictionary = json.load(f)
                logger.info(f"Missing Data Stats:\n{json.dumps(missing_stats_dictionary, indent=4)}")

            logger.success(f"Loaded {data_label} RBP PPI events and controls.")

        else: 
            
            logger.info("NO CACHE... Hence, retrieving RBP PPI events and controls for both cell lines.")

            if not hasattr(self, 'linear_model_results'):
                self.load_linear_model_results()

            if not hasattr(self, 'shap_data'):
                self.load_SHAP_data()

            missing_data_stats = {}

            for cell_line in self.cell_lines: 

                logger.info(f"Retrieving RBP PPI events and controls for {cell_line}.")

                input_data = self.shap_data[cell_line].select(self.binding_columns[cell_line] + ["graph_index"])

                rbp_pair_position_combinations = {(rbp_ppi_pair, position_key): None for rbp_ppi_pair in self.rbp_ppi[cell_line] for position_key in self.splice_junction_position_renaming.keys()}
                assert len(rbp_pair_position_combinations) == len(self.rbp_ppi[cell_line]) * len(self.splice_junction_position_renaming.keys())

                no_ppi_combinations = []
                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.get_number_SLURM_CPUs()) as executor:
                    same_pos_ppi = {
                        executor.submit(self.retrieve_rbp_cobinding, input_data, rbp_ppi_pair, position_key, position_key): (rbp_ppi_pair, position_key) for rbp_ppi_pair, position_key in rbp_pair_position_combinations.keys()
                    }

                    for future in tqdm.tqdm(concurrent.futures.as_completed(same_pos_ppi), total=len(same_pos_ppi), desc="Same Position PPI Events"):
                        result = future.result()

                        if result.shape[0] >= MIN_EXAMPLES_THRESHOLD:
                            results.append(result)
                        else:
                            rbp_pair_position_combinations.pop(same_pos_ppi[future])
                            no_ppi_combinations.append(same_pos_ppi[future])
                
                same_pos_ppi = pl.concat(results, how="vertical")
                assert same_pos_ppi.is_duplicated().any() == False
                
                no_single_binders = []
                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.get_number_SLURM_CPUs()) as executor:
                    single_binders_futures = {
                        executor.submit(self.retrieve_single_binders, input_data, rbp_ppi_pair, position_key): (rbp_ppi_pair, position_key) for rbp_ppi_pair, position_key in rbp_pair_position_combinations.keys()
                    }

                    for future in tqdm.tqdm(concurrent.futures.as_completed(single_binders_futures), total=len(single_binders_futures), desc="Single Binder Events"):
                        result = future.result()
                        if not result.is_empty():
                            results.append(result)
                        else: 
                            no_single_binders.append(single_binders_futures[future])

                single_binders = pl.concat(results, how="vertical")
                assert single_binders.is_duplicated().any() == False
                assert len(no_single_binders) == 0

                no_examples_data = []

                for no_example_type, combinations in [("No PPI Examples", no_ppi_combinations)]:
                    for rbp_ppi_pair, position_key in combinations:
                        no_examples_data.append({
                            "No Examples Type": no_example_type,
                            "RBP Pair": f"{rbp_ppi_pair[0].upper()}-{rbp_ppi_pair[1].upper()}",
                            "Position": self.splice_junction_position_renaming[position_key]
                        })

                no_examples_df = pd.DataFrame(no_examples_data).sort_values(by=["No Examples Type", "RBP Pair", "Position"])

                output_file = f"{PPI_MISSING_SUMMARY_DIR}/{cell_line}_{self.distance_threshold}.tsv"
                no_examples_df.to_csv(output_file, sep="\t", index=False)

                final_data = pl.concat(
                    [same_pos_ppi, single_binders], 
                    how="diagonal"
                ).sort("graph_index")
                assert final_data.is_duplicated().any() == False
                
                possible_combinations = len(self.rbp_ppi[cell_line]) * len(self.splice_junction_position_renaming.keys())
                omitted_combinations = pl.from_pandas(no_examples_df).select(["RBP Pair", "Position"]).n_unique()
                final_combinations = final_data.select(["RBP Pair", "Position"]).n_unique()

                print(f"{cell_line} -- Possible Combinations: {possible_combinations} | Omitted Combinations: {omitted_combinations} | Final Combinations: {final_combinations}")
                assert final_combinations == possible_combinations - omitted_combinations

                missing_data_stats[cell_line] = {}
                missing_data_stats[cell_line]["Possible Combinations"] = possible_combinations
                missing_data_stats[cell_line]["Omitted Combinations"] = omitted_combinations
                missing_data_stats[cell_line]["Final Combinations"] = final_combinations
                
                self._cache_to_featherv2(final_data, f"{self.PPI_CACHE_DIR}/PPI-ASSIGNMENTS-{cell_line}-{self.distance_threshold}.feather")

                for model_type, df in [("linear", self.linear_model_results[cell_line]), ("xgboost", self.shap_data[cell_line])]:

                    df = df.join(final_data, on="graph_index", how="left", validate="1:m")
                    df = df.sort("graph_index")

                    self._cache_to_featherv2(df, f"{self.PPI_CACHE_DIR}/{cell_line}-{self.distance_threshold}-{model_type}-ppi_events_and_controls.feather")
        
            with open(f"{PPI_MISSING_SUMMARY_DIR}/missing_data_stats_{self.distance_threshold}.json", "w") as f:
                json.dump(missing_data_stats, f, indent=4)

            logger.success("Finished retrieving and caching RBP PPI events and controls.")


    def plot_same_pos_ppi_per_pair_combination(self): 
        logger.info("Plotting number of same position PPI rows per pair combination.")

        swarmplot_df = []
        for cell_line in self.cell_lines: 

            data = self.xgboost_ppi[cell_line]
            assert all(data["Data Partition"] == "test"), logger.error(f"Not all values in 'Data Partition' column are 'test' for {cell_line}.")
            data = data.filter(pl.col("PPI Analysis Category") == "Same Pos. PPI")

            grouped_data = data.group_by(["RBP Pair", "Position"]).agg(pl.count()).to_pandas()
            grouped_data["Cell Line"] = cell_line
            swarmplot_df.append(grouped_data)

        swarmplot_df = pd.concat(swarmplot_df).sort_values(by='count', ascending=False)

        plt.figure(figsize=(6,3), dpi=200)

        sns.swarmplot(x="Cell Line", y="count", data=swarmplot_df, order=["K562", "HepG2"], palette=["tomato", "skyblue"], size=4)
        sns.boxplot(x="Cell Line", y="count", data=swarmplot_df, order=["K562", "HepG2"], width=0.3, showcaps=False, boxprops={'facecolor':'None', 'edgecolor':'black'}, flierprops={'marker': 'o', 'markersize': 2, 'markerfacecolor': 'black'}, medianprops={'color': 'black'}, whiskerprops={'color': 'black'}, capprops={'color': 'black'})

        plt.ylabel("# PPI Graphs")
        plt.title("Number of PPI Examples per Pair-Position Combination")
        plt.show()

    
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
                spearman_corr, _ = scipy.stats.spearmanr(data["Total Binding"], data["target"])

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
                spearman_corr, _ = scipy.stats.spearmanr(data["Total Binding"], data["target"])

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

    
    def calculate_summary_PPI_performance(self): 
        
        SUMMARY_PPI_PERFORMANCE_OUTPUT_FILE = f"../output/ppi_summary_stats/summary_ppi_performance_{self.distance_threshold}.tsv"

        if pathlib.Path(SUMMARY_PPI_PERFORMANCE_OUTPUT_FILE).exists():

            self.summary_ppi_df = pd.read_csv(SUMMARY_PPI_PERFORMANCE_OUTPUT_FILE, sep="\t")
            logger.success("FROM CACHE: loaded summary PPI performance.")

        else: 
            
            if not hasattr(self, 'linear_ppi') or not hasattr(self, 'xgboost_ppi'):
                self.retrieve_rbp_ppi_events_and_controls()
            
            logger.info("Calculating summary PPI performance.")

            summary_data = []

            for cell_line in self.cell_lines:
                for model in ["xgboost", "linear"]:
                    data = getattr(self, f"{model}_ppi")[cell_line]

                    assert data["Data Partition"].unique().to_list() == ["test"], logger.error(f"Data Partition column does not contain 'test' for {cell_line} - {model}.")

                    for category in ["Same Pos. PPI", "Single Binders", "Neither"]:
                        if category == "Same Pos. PPI":
                            subset = data.filter(pl.col("PPI Analysis Category") == category)
                        elif category == "Single Binders":
                            same_pos_ppi_graph_indices = data.filter(pl.col("PPI Analysis Category") == "Same Pos. PPI").select("graph_index").unique()
                            subset = data.filter((pl.col("PPI Analysis Category").str.ends_with(" Only")) & (~pl.col("graph_index").is_in(same_pos_ppi_graph_indices["graph_index"])))
                        elif category == "Neither":
                            subset = data.filter(pl.col("PPI Analysis Category").is_null())

                        subset = subset.select(["graph_index", "psi_hat", "target"]).unique()

                        assert subset["graph_index"].n_unique() == subset.shape[0], logger.error(f"Duplicate values found in 'graph_index' for {cell_line} in {category} category.")
                        assert subset.is_empty() == False, logger.error(f"No data found for {cell_line} - {model} - {category}.")

                        r2 = r2_score(subset["target"], subset["psi_hat"])
                        summary_data.append(
                            [ 
                                cell_line, 
                                model, 
                                category, 
                                r2, 
                                subset.shape[0], 
                                (subset.shape[0] / data["graph_index"].n_unique()) * 100
                            ]
                        )

            summary_df = pd.DataFrame(summary_data, columns=["Cell Line", "Model", "PPI Category", "R2 Value", "# Rows", "% Dataset"]).sort_values("R2 Value", ascending=False)
            summary_df["PPI Category"] = summary_df["PPI Category"].replace({"Same Pos. PPI": ">= 1 Same Pos. PPI"})
            summary_df.to_csv(SUMMARY_PPI_PERFORMANCE_OUTPUT_FILE, sep="\t", index=False)

            logger.success("Summary PPI performance calculated and saved.")


    def plot_summary_PPI_performance(self):

        if not hasattr(self, 'summary_ppi_df'):
            self.calculate_summary_PPI_performance()

        logger.info("Plotting PPI performance.")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, dpi=200)

        categories = [">= 1 Same Pos. PPI", "Single Binders", "Neither"]
        models = ["xgboost", "linear"]
        colors = {"xgboost": "gold", "linear": "darkolivegreen"}
        width = 0.35
        lowest_r2_value = 0 
        highest_r2_value = 0 


        for ax, cell_line in zip(axes, self.cell_lines):
            subset_df = self.summary_ppi_df[self.summary_ppi_df["Cell Line"] == cell_line]

            r2_values = {model: [] for model in models}
            counts = {model: [] for model in models}
            percentages = {model: [] for model in models}
            labels = []

            for category in categories:
                labels.append(category)
                for model in models:
                    data = subset_df[(subset_df["PPI Category"] == category) & (subset_df["Model"] == model)]
                    assert len(data) == 1, logger.error(f"Multiple rows found for {cell_line} - {category} - {model} in dataset.")

                    r2_values[model].append(data["R2 Value"].values[0])
                    counts[model].append(data["# Rows"].values[0])
                    percentages[model].append(data["% Dataset"].values[0])

            x = range(len(categories))

            for i, model in enumerate(models):
                if model == "xgboost":
                    label = "XGBoost"
                elif model == "linear":
                    label = "OLS Lin. Reg."

                ax.bar([p + i * width for p in x], r2_values[model], width=width, color=colors[model], edgecolor="black", label=label)

                for j, (r2_value, pct) in enumerate(zip(r2_values[model], percentages[model])):
                    r2_offset = 0.07
                    percent_data_offset = 0.02

                    if r2_value < 0:
                        ax.annotate(f'{r2_value:.2f}', (j + i * width, r2_value-r2_offset), ha='center', va='top', color='red', fontsize=12)
                        ax.annotate(f'{pct:.1f}', (j + i * width, r2_value-percent_data_offset), ha='center', va='top', color='blue', fontsize=12)
                    else:
                        ax.annotate(f'{r2_value:.2f}', (j + i * width, r2_value+r2_offset), ha='center', va='bottom', color='red', fontsize=12)
                        ax.annotate(f'{pct:.1f}', (j + i * width, r2_value+percent_data_offset), ha='center', va='bottom', color='blue', fontsize=12)

            ax.set_xticks([p + width / 2 for p in x])
            ax.set_xticklabels(labels, ha='center')
            ax.set_title(f"{cell_line}", fontsize=16)

            min_r2_value = min(r2_values["xgboost"] + r2_values["linear"])
            if min_r2_value < lowest_r2_value:
                lowest_r2_value = min_r2_value

            max_r2_value = max(r2_values["xgboost"] + r2_values["linear"])
            if max_r2_value > highest_r2_value:
                highest_r2_value = max_r2_value

        if lowest_r2_value<0: 
            y_lim_low_end = lowest_r2_value - 0.15
        else:
            y_lim_low_end = 0

        ax.set_ylim(y_lim_low_end, highest_r2_value + 0.25)

        for ax in axes:
            ax.text(0.62, 0.9, "% of Data", ha='center', va='bottom', color='blue', fontsize=14, transform=ax.transAxes)
            ax.text(0.5, 0.9, "--", ha='center', va='bottom', color='black', fontsize=14, transform=ax.transAxes)
            ax.text(0.39, 0.9, "R2 Score", ha='center', va='bottom', color='red', fontsize=14, transform=ax.transAxes)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        fig.supxlabel("PPI Category", fontsize=20)
        fig.supylabel("R2 Score", fontsize=20)
        plt.suptitle(f"Test Data: R2 Scores for PPI Categories per Model\nNOTE: each row goes to ONLY 1 category", fontsize=20)

        plt.tight_layout()
        plt.show()        


    def calculate_pair_position_combination_r2_scores(self): 
            
        PAIR_POSITION_R2_SCORES_OUTPUT_FILE = f"../output/ppi_summary_stats/pair_position_r2_scores_{self.distance_threshold}.tsv"

        if pathlib.Path(PAIR_POSITION_R2_SCORES_OUTPUT_FILE).exists():

            self.pair_position_r2_scores = pd.read_csv(PAIR_POSITION_R2_SCORES_OUTPUT_FILE, sep="\t")

            # Assert that each unique combination of "Dataset", "Cell Line", "Model", "RBP Pair", "Position" columns is two total rows long
            assert self.pair_position_r2_scores.groupby(["Cell Line", "Model", "RBP Pair", "Position"]).size().eq(3).all(), "Each unique combination of 'Cell Line', 'Model', 'RBP Pair', 'Position' should have exactly 3 rows."

            # Assert that there is at least one row for every combination of "Dataset", "Cell Line", and "Model" columns
            assert self.pair_position_r2_scores.groupby(["Cell Line", "Model"]).size().ge(2).all(), "There should be at least 2 rows for every combination of 'Cell Line' and 'Model'."

            logger.success("FROM CACHE: loaded pair position R2 scores.")

        else: 

            if not hasattr(self, 'linear_ppi') or not hasattr(self, 'xgboost_ppi'):
                self.retrieve_rbp_ppi_events_and_controls()

            logger.info("NO CACHE... Hence, calculating pair position combination R2 scores.")

            pair_position_r2_scores = []

            for cell_line in self.cell_lines:
                for model in ["xgboost", "linear"]:
                    data = getattr(self, f"{model}_ppi")[cell_line]
                    assert data["Data Partition"].unique().to_list() == ["test"], logger.error(f"Data Partition column does not contain 'test' for {cell_line} - {model}.")

                    for (rbp_pair, position), group in data.group_by(["RBP Pair", "Position"]):

                        if rbp_pair is not None:
                            same_pos_ppi_group = group.filter(pl.col("PPI Analysis Category") == "Same Pos. PPI")
                            single_binders_group = group.filter(pl.col("PPI Analysis Category").str.ends_with(" Only"))
                            neither = data.filter(
                                (pl.col("RBP Pair") != rbp_pair) |
                                (pl.col("Position") != position)
                            )
                        
                            assert len(same_pos_ppi_group) >=3 and len(single_binders_group) >=3 and len(neither)>=3, logger.error(f"Insufficient data for {cell_line} - {model} - {rbp_pair} - {position}.")

                            for category, df in [("Same Pos. PPI", same_pos_ppi_group), ("Single Binders", single_binders_group), ("Neither", neither)]:
                            
                                subset = df.select(["graph_index", "psi_hat", "target"])
                                
                                if category == "Neither":
                                    subset = subset.unique()

                                assert subset["graph_index"].n_unique() == subset.shape[0], logger.error(f"Duplicate values found in 'graph_index' for {cell_line} in {category} category for {rbp_pair} at position {position}.")

                                r2 = r2_score(subset["target"], subset["psi_hat"])
                                percent_rows = (subset.shape[0] / data["graph_index"].n_unique()) * 100

                                pair_position_r2_scores.append([cell_line, model, rbp_pair, position, category, r2, subset.shape[0], percent_rows])

            self.pair_position_r2_scores = pd.DataFrame(pair_position_r2_scores, columns=["Cell Line", "Model", "RBP Pair", "Position", "PPI Category", "R2 Value", "# Unique Graphs", "% Unique Graphs"]).sort_values("R2 Value", ascending=False)
            self.pair_position_r2_scores.to_csv(PAIR_POSITION_R2_SCORES_OUTPUT_FILE, sep="\t", index=False)

            logger.success("Pair position combination R2 scores calculated and saved.")


    def plot_global_pair_position_combination_r2_scores(self):
            
        if not hasattr(self, 'pair_position_r2_scores'):
            self.calculate_pair_position_combination_r2_scores()

        MIN_VALUE = -0.1
        xgboost_data = self.pair_position_r2_scores[self.pair_position_r2_scores["Model"] == "xgboost"].copy(deep=True)
        xgboost_data.loc[xgboost_data["R2 Value"] < MIN_VALUE, "R2 Value"] = MIN_VALUE

        title_addendum = f"NOTE: all R2 score values < {MIN_VALUE} were made {MIN_VALUE}"
        
        logger.info("Comparing relationship between number of rows and R2 scores for each pair-position combination.")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True, dpi=200)

        for ax, cell_line in zip(axes, self.cell_lines):
            data = xgboost_data[xgboost_data["Cell Line"] == cell_line]
            hue_order = ["Same Pos. PPI", "Single Binders", "Neither"]

            sns.scatterplot(x="# Unique Graphs", y="R2 Value", hue="PPI Category", hue_order = hue_order, data=data, ax=ax, palette="Set2", edgecolor="black", s=20,)
            assert data.groupby(["RBP Pair", "Position", "Model"]).size().eq(3).all(), print(data.groupby(["RBP Pair", "Position"]).size())

            num_pairs = data.groupby(["RBP Pair", "Position"]).ngroups
            spearman_corr, _ = scipy.stats.spearmanr(data["# Unique Graphs"], data["R2 Value"])

            ax.text(0.5, 0.5, f"# Pair-Position Combos: {num_pairs}\nSpearman r: {spearman_corr:.2f}", 
                    transform=ax.transAxes, verticalalignment='center', horizontalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

            ax.set_title(f"{cell_line}", fontsize=16)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.legend().set_visible(False)

        fig.suptitle(f"XGBoost Test Set: R2 Scores vs. # Rows per Pair-Position Combination\n{title_addendum}", fontsize=20, y=0.98)
        fig.supxlabel("# Rows", fontsize=18)
        fig.supylabel("R2 Score", fontsize=18)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        logger.success("Plotted relationship between number of rows and R2 scores for each pair-position combination.")

        logger.info("Plotting pair position combination R2 scores.")

        fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True, sharex=True, dpi=200)

        for ax, cell_line in zip(axes, self.cell_lines):
            data = xgboost_data[xgboost_data["Cell Line"] == cell_line]
            assert data.groupby(["RBP Pair", "Position", "Model"]).size().eq(3).all(), print(data.groupby(["RBP Pair", "Position"]).size())

            x_axis_order = ["Same Pos. PPI", "Single Binders", "Neither"]
            sns.swarmplot(
                x="PPI Category", y="R2 Value", data=data, ax=ax, palette="Set2", order=x_axis_order, edgecolor="black", size=1,
            )
            sns.boxplot(
                x="PPI Category", y="R2 Value", data=data, ax=ax, palette="Set2", order=x_axis_order, 
                boxprops={'facecolor':'None', 'linewidth': 2},
                whiskerprops={'linewidth': 2},
                capprops={'linewidth': 2},
                medianprops={'linewidth': 2},
                flierprops={'marker': 'o', 'markersize': 5, 'linestyle': 'none'}
            )

            unique_groups = data.groupby(["RBP Pair", "Position"]).ngroups
            total_points = data.groupby(["Model", "PPI Category"]).size().reset_index(name="Count")
            assert len(total_points["Count"].unique()) == 1 and total_points["Count"].unique()[0]== unique_groups, logger.error(f"Multiple counts found for {cell_line}.")
            
            ax.set_title(f"{cell_line}", fontsize=16)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.legend().set_visible(False)

            ax.tick_params(axis='x', labelsize=16)
            
            ax.text(0.83, 0.9, f"# Pair-Position Combos: {unique_groups}", 
                    transform=ax.transAxes, verticalalignment='top', horizontalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
            
            top_5_points = data[data["PPI Category"] == "Same Pos. PPI"].nlargest(5, "R2 Value")
            for i, (_, row) in enumerate(top_5_points.iterrows()):
                ax.text(0.45, 0.95 - i * 0.05, f"{row['RBP Pair']} Pos. {row['Position']}: ", 
                        ha='right', va='center', fontsize=12, color='black', transform=ax.transAxes
                    )
                ax.text(0.45, 0.95 - i * 0.05, f"{row['R2 Value']:.2f}", 
                        ha='left', va='center', fontsize=12, color='blue', transform=ax.transAxes
                    )
                ax.text(0.5, 0.95 - i * 0.05, f"; {row['# Unique Graphs']} Graphs", 
                        ha='left', va='center', fontsize=12, color='red', transform=ax.transAxes
                    )
                
        fig.suptitle("XGBoost Test Dataset: R2 Scores per Pair-Position Combination\nNOTE: All R2 values < 0 set to 0", fontsize=24, y=0.98)
        fig.supxlabel("PPI Category", fontsize=22)
        fig.supylabel("R2 Score", fontsize=22, x=-0.01)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        logger.success("Plotted pair position combination R2 scores.")
    

    def compare_ppi_vs_single_binder_r2_table(self): 

        PPI_VS_SINGLE_BINDER_OUTPUT_FILE = f"../output/ppi_summary_stats/pair_position_ppi_vs_single_binder_r2_scores_{self.distance_threshold}.tsv"

        if pathlib.Path(PPI_VS_SINGLE_BINDER_OUTPUT_FILE).exists():
                
                self.ppi_vs_single_binder_df = pd.read_csv(PPI_VS_SINGLE_BINDER_OUTPUT_FILE, sep="\t")
                logger.success("FROM CACHE: loaded PPI vs. Single Binder R2 scores.")

                return self.ppi_vs_single_binder_df.head()
        
        else: 
            
            if not hasattr(self, 'pair_position_r2_scores'):
                self.calculate_pair_position_combination_r2_scores()

            logger.info("Creating table comparing PPI vs. Single Binder R2 scores.")

            xgboost_data = self.pair_position_r2_scores[self.pair_position_r2_scores["Model"] == "xgboost"]

            new_table = []

            for cell_line in self.cell_lines:
                cell_line_data = xgboost_data[xgboost_data["Cell Line"] == cell_line]

                grouped_data = cell_line_data.groupby(["RBP Pair", "Position"])
                assert grouped_data.size().eq(3).all(), "Each unique combination of 'RBP Pair' and 'Position' should have exactly 3 rows."

                for (rbp_pair, position), group in grouped_data:
                    same_pos_ppi_r2 = group[group["PPI Category"] == "Same Pos. PPI"]
                    assert len(same_pos_ppi_r2) == 1, logger.error(f"Multiple rows found for {cell_line} - {rbp_pair} - {position} in 'Same Pos. PPI' category.")

                    single_binders_r2 = group[group["PPI Category"] == "Single Binders"]
                    assert len(single_binders_r2) == 1, logger.error(f"Multiple rows found for {cell_line} - {rbp_pair} - {position} in 'Single Binders' category.")

                    assert group["Model"].nunique() == 1 and group["Model"].unique()[0] == "xgboost", logger.error(f"Model column values are not all 'xgboost' for {cell_line} - {rbp_pair} - {position}.")
                    
                    new_table.append(
                        {
                            "Cell Line": cell_line,
                            "Model": "xgboost",
                            "RBP Pair": rbp_pair,
                            "Position": position,
                            "Same Pos. PPI R2": same_pos_ppi_r2["R2 Value"].values[0],
                            "Single Binders R2": single_binders_r2["R2 Value"].values[0],
                            "# Graphs (Same Pos. PPI)": same_pos_ppi_r2["# Unique Graphs"].values[0],
                            "# Graphs (Single Binders)": single_binders_r2["# Unique Graphs"].values[0],
                            "Difference": same_pos_ppi_r2["R2 Value"].values[0] - single_binders_r2["R2 Value"].values[0]
                        }
                    )

            new_table_df = pd.DataFrame(new_table).sort_values("Difference", ascending=False)
            assert new_table_df.groupby(["Cell Line", "Model", "RBP Pair", "Position"]).size().eq(1).all(), "Each unique combination of 'Cell Line', 'Model', 'RBP Pair', 'Position' should have exactly 1 row."
            
            new_table_df.to_csv(PPI_VS_SINGLE_BINDER_OUTPUT_FILE, sep="\t", index=False)

            logger.success("PPI vs. Single Binder R2 scores calculated and saved.")


    def plot_pair_position_category_line_r2_scores(self):

        if not hasattr(self, 'ppi_vs_single_binder_df'):
            self.compare_ppi_vs_single_binder_r2_table()

        logger.info("Plotting PPI vs. Single Binder R2 scores connected lines plot.")

        original_data = self.ppi_vs_single_binder_df.copy(deep=True)
        assert original_data["Model"].nunique() == 1 and original_data["Model"].unique()[0] == "xgboost", logger.error("Model column values are not all 'xgboost'.")
        
        MIN_VALUE = -0.1
        original_data.loc[original_data["Same Pos. PPI R2"] < MIN_VALUE, "Same Pos. PPI R2"] = MIN_VALUE
        original_data.loc[original_data["Single Binders R2"] < MIN_VALUE, "Single Binders R2"] = MIN_VALUE
        title_addendum = f"NOTE: all R2 score values < {MIN_VALUE} were set to {MIN_VALUE}"

        fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=200, sharex=True, sharey=True)

        for ax, cell_line in zip(axes, self.cell_lines):
            data = original_data[original_data["Cell Line"] == cell_line]

            for _, row in data.iterrows():
                ax.plot(["Same Pos. PPI", "Single Binders"], [row["Same Pos. PPI R2"], row["Single Binders R2"]], marker='o', markerfacecolor='none', markeredgecolor='black')

                ax.set_title(f"{cell_line}", fontsize=22)
                ax.set_xlabel("")
                ax.set_ylabel("")

            top_5_rows = data.nlargest(5, "Difference")
            for i, (_, row) in enumerate(top_5_rows.iterrows()):
                ax.text(0.55, 0.95 - i * 0.05, f"{row['RBP Pair']} Pos. {row['Position']}:", 
                        ha='right', va='center', fontsize=12, color='red', transform=ax.transAxes)
                ax.text(0.55, 0.95 - i * 0.05, f" {row['Difference']:.2f}", 
                        ha='left', va='center', fontsize=12, color='blue', transform=ax.transAxes)

        fig.supxlabel("PPI Category", fontsize=20)
        fig.supylabel("R2 Score", fontsize=20, x=0.01)
        fig.text(0.51, 0.88, "Pair-Position", fontsize=16, ha='center',va='center', color='red')
        fig.text(0.51, 0.83, "(Same Pos. R2) - (Single R2)", ha='center', va='center', fontsize=16, color='blue')
        plt.suptitle(f"XGBoost Test R2 Scores for Same Pos. PPI vs. Single Binders\n{title_addendum}", fontsize=26, y=1.03)

        plt.tight_layout()
        plt.show()


    def create_and_plot_num_bindings_vs_r2_score(self): 

        if not hasattr(self, 'ppi_vs_single_binder_df'):
            self.create_ppi_vs_single_binder_table()
        
        for _, row in self.ppi_vs_single_binder_df.iterrows():
            cell_line = row["Cell Line"]
            model = row["Model"]
            rbp_pair = row["RBP Pair"]
            position = row["Position"]

            data = getattr(self, f"{model}_ppi")[cell_line]
            
            subset_data = data.filter(
                (pl.col("RBP Pair") == rbp_pair) & 
                (pl.col("Position") == str(position))
            )

            assert subset_data["Data Partition"].unique().to_list() == ["test"], logger.error(f"Data Partition column does not contain 'test' for {cell_line} - {model} - {rbp_pair} - {position}.")
            unique_total_bindings = sorted(subset_data["Total Binding"].unique().to_list())
            grouped_bindings = []
            for i in range(0, 15, 3):
                grouped_bindings.append(f"{i+1}-{i+3}")
            grouped_bindings.append(">15")

            r2_scores = {"Same Pos. PPI": [], "Single Binders": []}
            binding_counts = []
            num_rows = {"Same Pos. PPI": [], "Single Binders": []}

            for group in grouped_bindings:
                if group == ">15":
                    filtered_data = subset_data.filter(pl.col("Total Binding") > 15)
                else:
                    lower, upper = map(int, group.split('-'))
                    filtered_data = subset_data.filter((pl.col("Total Binding") >= lower) & (pl.col("Total Binding") <= upper))

                for category in ["Same Pos. PPI", "Single Binders"]:
                    if category == "Same Pos. PPI":
                        category_data = filtered_data.filter(pl.col("PPI Analysis Category") == category)
                    elif category == "Single Binders":
                        category_data = filtered_data.filter(pl.col("PPI Analysis Category").str.contains(" Only"))
                    
                    if not category_data.is_empty() and category_data.shape[0] >= 3:
                        assert category_data["graph_index"].n_unique() == category_data.shape[0], logger.error(f"Duplicate values found in 'graph_index' for {cell_line} in {category} category for {rbp_pair} at position {position}.")
                        r2 = r2_score(category_data["target"], category_data["psi_hat"])
                        if r2 < -5: 
                            r2=0
                        r2_scores[category].append(r2)
                        num_rows[category].append(category_data.shape[0])
                    else:
                        r2_scores[category].append(None)
                        num_rows[category].append(0)

                binding_counts.append(group)

            plt.figure(figsize=(10, 4))
            for category in ["Same Pos. PPI", "Single Binders"]:
                plt.plot(binding_counts, r2_scores[category], marker='o', label=category)
                for i, (x, y) in enumerate(zip(binding_counts, r2_scores[category])):
                    if y is not None:
                        plt.text(x, y, str(num_rows[category][i]), fontsize=12, color='green', ha='center', va='bottom')

                # Calculate and add Spearman correlation
                valid_indices = [i for i, y in enumerate(r2_scores[category]) if y is not None]
                if valid_indices:
                    valid_r2_scores = [r2_scores[category][i] for i in valid_indices]
                    valid_binding_counts = [binding_counts[i] for i in valid_indices]
                    spearman_corr, _ = scipy.stats.spearmanr(valid_binding_counts, valid_r2_scores)
                    y_offset = 0.5 + (0.05 * list(r2_scores.keys()).index(category))
                    plt.text(0.5, y_offset, f"{category} Spearman r: {spearman_corr:.2f}", transform=plt.gca().transAxes, fontsize=12, color='blue', ha='center', va='center')

            plt.xlabel("Number of Bindings")
            plt.ylabel("R2 Score")
            plt.title(f"R2 Score vs. Number of Bindings for {cell_line} - {model} - {rbp_pair} - {position}")
            plt.legend()
            plt.grid(True)
            plt.show()


    def plot_pair_position_performance_and_local_shap(self):

        for self.cell_line in self.cell_lines:
            original_data = self.xgboost_ppi[self.cell_line].filter(pl.col("RBP Pair").is_not_null())
            assert all(original_data["Data Partition"] == "test"), logger.error(f"Not all values in 'Data Partition' column are 'test' for {self.cell_line}.")

            for (rbp_pair, position), data in original_data.group_by(["RBP Pair", "Position"]):
                logger.info(f"{rbp_pair} @ Pos. {position}: plotting performance and local SHAP values.")

                category_order = ["Same Pos. PPI"] + sorted([ category for category in data["PPI Analysis Category"].unique().to_list() if category.endswith(" Only")])
                colors = ["mediumorchid", "deepskyblue", "tomato"]
            
                for add_local_shap in [True, False]:

                    fig, axes = plt.subplots(2, 3, figsize=(18, 8), dpi=300,)

                    for i, (category, color) in enumerate(zip(category_order, colors)):
                        subset = data.filter(pl.col("PPI Analysis Category") == category).to_pandas()
                        axes[0][i].scatter(subset["target"], subset["psi_hat"], color=color, label=category, alpha=0.05, s=10,)
                        
                        axes[0][i].set_xlim(0, 1)
                        axes[0][i].set_ylim(0, 1)

                        axes[0][i].plot([0, 1], [0, 1], linestyle=':', color='gold', linewidth=2)

                        r2 = r2_score(subset["target"], subset["psi_hat"])
                        num_points = len(subset)
                        avg_target = subset["target"].mean()

                        axes[0][i].text(0.9, 0.09, f"Mean(Actual PSI): {avg_target:.2f}", transform=axes[0][i].transAxes, verticalalignment='top', horizontalalignment='right', fontsize=12)
                        axes[0][i].text(0.9, 0.17, f"R2: {r2:.2f}", transform=axes[0][i].transAxes, verticalalignment='top', horizontalalignment='right', fontsize=12)
                        axes[0][i].text(0.9, 0.25, f"# Points: {num_points}", transform=axes[0][i].transAxes, verticalalignment='top', horizontalalignment='right', fontsize=12)

                        axes[0][i].set_title(category, fontsize=16)
                        axes[0][i].set_xlabel("Actual", fontsize=14)
                        axes[0][i].set_ylabel("Predicted", fontsize=14)

                    plotting_df = []

                    for category in category_order:
                        subset = data.filter(pl.col("PPI Analysis Category") == category).to_pandas()

                        two_shap_cols = [f"{rbp}_{self.position_inverted_dict[position]}_shap" for rbp in rbp_pair.split("-")]
                        subset = subset[two_shap_cols]

                        if add_local_shap:
                            subset["Local SHAP"] = subset[two_shap_cols].sum(axis=1)

                            plotting_df.append(
                                pd.DataFrame(
                                    {
                                        "Summed Local SHAP": subset["Local SHAP"],
                                        "PPI Analysis Category": category
                                    }
                                )
                            )
                        
                        elif not add_local_shap: 

                            for rbp in rbp_pair.split("-"):
                                shap_col = f"{rbp}_{self.position_inverted_dict[position]}_shap"

                                plotting_df.append(
                                    pd.DataFrame(
                                        {
                                            "Local SHAP": subset[shap_col],
                                            "PPI Analysis Category": category,
                                            "RBP": rbp
                                        }
                                    )
                                )

                    plotting_df = pd.concat(plotting_df, ignore_index=True)
                    
                    if add_local_shap:
                        y="Summed Local SHAP"
                        hue=None
                        palette = colors
                        gap=0
                        subplot_title = "Summed Local SHAP per PPI Category"
                        plot_file_suffix= "summed_local_shap"
                    
                    elif not add_local_shap: 
                        y="Local SHAP"
                        hue="RBP"
                        palette = ["deepskyblue", "tomato"]
                        gap=0.3
                        subplot_title = "Local SHAP per PPI Category and RBP"
                        plot_file_suffix= "local_shap"

                    sns.violinplot(x="PPI Analysis Category", y=y, hue=hue, data=plotting_df, ax=axes[1][0], order=category_order, palette=palette, density_norm='width', gap=gap)

                    axes[1][0].axhline(0, color='lime', linestyle=':', linewidth=2)
                    axes[1][0].set_title(f"{subplot_title}", fontsize=20)
                    axes[1][0].set_xlabel("PPI Analysis Category", fontsize=14)
                    axes[1][0].set_ylabel(y, fontsize=14)

                    if not add_local_shap: 
                        axes[1][0].legend(bbox_to_anchor=(0.72, 0.95), fontsize=14)

                    # Remove the other two axes in the second row
                    fig.delaxes(axes[1][1])
                    fig.delaxes(axes[1][2])

                    # Adjust the layout to make the single plot span the entire row
                    axes[1][0].set_position([0.1, 0.05, 0.8, 0.35])

                    fig.suptitle(f"(XGBoost Test) {self.cell_line}: {rbp_pair} @ Pos. {position}", fontsize=24, y=1)

                    plt.tight_layout()
                    plt.savefig(
                        f"../output/performance_local_SHAP_plots/{self.cell_line}_{rbp_pair}_{position}_{plot_file_suffix}.png", 
                        dpi=200,
                        bbox_inches='tight', 
                    )
                    plt.close()

        logger.success("Plotted performance and local SHAP values for each pair-position combination in both cell lines.")


    def create_ppi_vs_single_binders_table(self): 

        PPI_VS_SINGLE_BINDERS_FILE=f"../output/ppi_summary_stats/summary_pair_position_metrics_{self.distance_threshold}.tsv"

        if pathlib.Path(PPI_VS_SINGLE_BINDERS_FILE).exists():
            self.ppi_vs_single_binders_df = pd.read_csv(PPI_VS_SINGLE_BINDERS_FILE, sep="\t")
            logger.success("FROM CACHE: loaded PPI vs. Single Binders table.")
            return self.ppi_vs_single_binders_df.head()

        else: 

            results = []

            for cell_line in self.cell_lines:
                original_data = self.xgboost_ppi[cell_line].filter(pl.col("PPI Analysis Category").is_not_null())
                assert all(original_data["Data Partition"] == "test"), logger.error(f"Not all values in 'Data Partition' column are 'test' for {cell_line}.")
                
                grouped_data = original_data.group_by(["RBP Pair", "Position"], maintain_order=True)
                
                for (rbp_pair, position), group in grouped_data:
                    group = group.with_columns(
                        pl.when(pl.col("PPI Analysis Category").str.ends_with(" Only"))
                        .then(pl.lit("Single Binders"))
                        .otherwise(pl.lit("Same Pos. PPI"))
                        .alias("PPI vs Single Binder")
                    )

                    same_pos_ppi = group.filter(pl.col("PPI vs Single Binder") == "Same Pos. PPI")
                    single_binders = group.filter(pl.col("PPI vs Single Binder") == "Single Binders")

                    ##############################
                    #### Actual PSI Values #######
                    ##############################
                    same_pos_ppi_mean = same_pos_ppi["target"].mean()
                    same_pos_ppi_std = same_pos_ppi["target"].std()

                    single_binders_mean = single_binders["target"].mean()
                    single_binders_std = single_binders["target"].std()

                    mean_diff = same_pos_ppi_mean - single_binders_mean
                    std_diff = same_pos_ppi_std - single_binders_std

                    ########################
                    #### Global SHAP #######
                    ########################
                    rbp1, rbp2 = rbp_pair.split("-")
                    shap_col1 = f"{rbp1}_{self.position_inverted_dict[position]}_shap"
                    shap_col2 = f"{rbp2}_{self.position_inverted_dict[position]}_shap"

                    same_pos_ppi_mean_shap1 = same_pos_ppi[shap_col1].abs().mean()
                    same_pos_ppi_mean_shap2 = same_pos_ppi[shap_col2].abs().mean()

                    single_binders_mean_shap1 = single_binders[shap_col1].abs().mean()
                    single_binders_mean_shap2 = single_binders[shap_col2].abs().mean()

                    mean_diff_shap1 = same_pos_ppi_mean_shap1 - single_binders_mean_shap1
                    mean_diff_shap2 = same_pos_ppi_mean_shap2 - single_binders_mean_shap2

                    #######################
                    #### Local SHAP #######
                    #######################
                    t_stat_shap1, t_p_value_shap1 = scipy.stats.ttest_ind(
                        same_pos_ppi[shap_col1].cast(pl.Float64), 
                        group.filter(pl.col("PPI Analysis Category") == f"{rbp1} Only")[shap_col1].cast(pl.Float64), 
                        equal_var=False
                    )
                    u_stat_shap1, u_p_value_shap1 = scipy.stats.mannwhitneyu(
                        same_pos_ppi[shap_col1].cast(pl.Float64), 
                        group.filter(pl.col("PPI Analysis Category") == f"{rbp1} Only")[shap_col1].cast(pl.Float64), 
                        alternative='two-sided'
                    )

                    t_stat_shap2, t_p_value_shap2 = scipy.stats.ttest_ind(
                        same_pos_ppi[shap_col2].cast(pl.Float64), 
                        group.filter(pl.col("PPI Analysis Category") == f"{rbp2} Only")[shap_col2].cast(pl.Float64), 
                        equal_var=False
                    )
                    u_stat_shap2, u_p_value_shap2 = scipy.stats.mannwhitneyu(
                        same_pos_ppi[shap_col2].cast(pl.Float64), 
                        group.filter(pl.col("PPI Analysis Category") == f"{rbp2} Only")[shap_col2].cast(pl.Float64), 
                        alternative='two-sided'
                    )

                    ###########################################
                    #### Welch's T-test & Mann Whitney U ######
                    ###########################################
                    t_stat, t_p_value = scipy.stats.ttest_ind(same_pos_ppi["target"].cast(pl.Float64), single_binders["target"].cast(pl.Float64), equal_var=False)
                    u_stat, u_p_value = scipy.stats.mannwhitneyu(same_pos_ppi["target"].cast(pl.Float64), single_binders["target"].cast(pl.Float64), alternative='two-sided')

                    results.append(
                        {
                            "Data Partition": "test",
                            "Model": "xgboost",
                            "Cell Line": cell_line,
                            "RBP Pair": rbp_pair,
                            "Position": position,
                            "Same Pos. PPI Rows": same_pos_ppi.shape[0],
                            "Single Binders Rows": single_binders.shape[0],
                            "Actual PSI Mean - PPI ": same_pos_ppi_mean,
                            "Actual PSI Mean - Single Binders": single_binders_mean,
                            "Actual PSI Mean Difference: PPI vs Single Binders": mean_diff,
                            "Actual PSI Std Dev - PPI": same_pos_ppi_std,
                            "Actual PSI Std Dev - Single Binders": single_binders_std,
                            "Actual PSI Std Dev Difference: PPI vs Single Binders": std_diff,
                            "Global SHAP RBP 1 - PPI": same_pos_ppi_mean_shap1,
                            "Global SHAP RBP 1 - Single Binders": single_binders_mean_shap1,
                            "Global SHAP RBP 1 Difference: PPI vs Single Binders": mean_diff_shap1,
                            "Global SHAP RBP 2 - PPI": same_pos_ppi_mean_shap2,
                            "Global SHAP RBP 2 - Single Binders": single_binders_mean_shap2,
                            "Global SHAP RBP 2 Difference: PPI vs Single Binders": mean_diff_shap2,
                            "Local SHAP: RBP 1 PPI vs RBP 1 Only Welch's T-test Stat": t_stat_shap1,
                            "Local SHAP: RBP 1 PPI vs RBP 1 Only Welch's T-test P-value": t_p_value_shap1,
                            "Local SHAP: RBP 1 PPI vs RBP 1 Only Mann Whitney U Stat": u_stat_shap1,
                            "Local SHAP: RBP 1 PPI vs RBP 1 Only Mann Whitney U P-value": u_p_value_shap1,
                            "Local SHAP: RBP 2 PPI vs RBP 2 Only Welch's T-test Stat": t_stat_shap2,
                            "Local SHAP: RBP 2 PPI vs RBP 2 Only Welch's T-test P-value": t_p_value_shap2,
                            "Local SHAP: RBP 2 PPI vs RBP 2 Only Mann Whitney U Stat": u_stat_shap2,
                            "Local SHAP: RBP 2 PPI vs RBP 2 Only Mann Whitney U P-value": u_p_value_shap2,
                            "PPI vs Single Binders PSI Welch's T-test Stat": t_stat,
                            "PPI vs Single Binders PSI Welch's T-test P-value": t_p_value,
                            "PPI vs Single Binders PSI Mann Whitney U Stat": u_stat,
                            "PPI vs Single Binders PSI Mann Whitney U P-value": u_p_value,
                        }
                    )

            results_df = pd.DataFrame(results).sort_values(["Data Partition", "Model", "Cell Line", "RBP Pair", "Position"])

            psi_welch_p_values = results_df["PPI vs Single Binders PSI Welch's T-test P-value"].tolist()
            psi_mannwhitney_p_values = results_df["PPI vs Single Binders PSI Mann Whitney U P-value"].tolist()

            # Apply FDR BH correction
            psi_welch_corrected_p_values = scipy.stats.false_discovery_control(psi_welch_p_values, method='bh')
            psi_mannwhitney_corrected_p_values = scipy.stats.false_discovery_control(psi_mannwhitney_p_values, method='bh')

            # Insert the corrected p-values next to their respective p-value columns
            results_df.insert(
                results_df.columns.get_loc("PPI vs Single Binders PSI Welch's T-test P-value") + 1,
                "PPI vs Single Binders PSI Welch's T-test FDR BH",
                psi_welch_corrected_p_values
            )
            results_df.insert(
                results_df.columns.get_loc("PPI vs Single Binders PSI Mann Whitney U P-value") + 1,
                "PPI vs Single Binders PSI Mann Whitney U FDR BH",
                psi_mannwhitney_corrected_p_values
            )
            
            null_counts = results_df.isnull().sum()
            print("Number of null values in each column:")
            print(null_counts[null_counts > 0])

            results_df.to_csv(PPI_VS_SINGLE_BINDERS_FILE, sep="\t", index=False)
            
            logger.success("PPI vs. Single Binders table created and saved.")
    

    def run_pair_position_ols_linear_regression(self): 

        OLS_LIN_REG_CACHE_FILE="../output/ols_lin_reg_ppi_results/ppi_ols_lin_reg_results.tsv"

        if pathlib.Path(OLS_LIN_REG_CACHE_FILE).exists():
            self.ppi_ols_lin_reg_results = pd.read_csv(OLS_LIN_REG_CACHE_FILE, sep="\t")
            logger.success("FROM CACHE: loaded OLS linear regression results.")
            return self.ppi_ols_lin_reg_results.head()
        

        else: 
            logger.info("NO CACHE... hence, running OLS linear regression per pair-position combo for coefficient testing using ALL data")

            unique_combinations = {}

            for cell_line in self.cell_lines:
                data = self.xgboost_ppi[cell_line]
                data = data.filter(pl.col("PPI Analysis Category").is_not_null())
                combinations = data.select(["RBP Pair", "Position"]).unique().sort(["RBP Pair", "Position"]).to_dict(as_series=False)
                unique_combinations[cell_line] = list(zip(combinations["RBP Pair"], combinations["Position"]))
        
            self.delete_PPI_data()
            self.retrieve_rbp_ppi_events_and_controls(test_only=False)

            results = []

            for cell_line in unique_combinations:
                for rbp_pair, position in unique_combinations[cell_line]:

                    rbp1, rbp2 = rbp_pair.split("-")
                    binding_col1 = f"{rbp1}_{self.position_inverted_dict[position]}"
                    binding_col2 = f"{rbp2}_{self.position_inverted_dict[position]}"
                    
                    subset_data = self.xgboost_ppi[cell_line].select(
                        ["target", binding_col1, binding_col2, "Data Partition", "graph_index"]
                    ).unique()   

                    assert subset_data["Data Partition"].n_unique() == 3, logger.error(f"Data Partition column does not contain 3 unique values for {cell_line} - {rbp_pair} - {position}.")
                    assert subset_data["graph_index"].n_unique() == subset_data.shape[0], logger.error(f"Duplicate values found in 'graph_index' for {cell_line} - {rbp_pair} - {position}.")

                    subset_data = subset_data.with_columns(
                        pl.when((pl.col(binding_col1) == 1) & (pl.col(binding_col2) == 1))
                        .then(pl.lit(1))
                        .otherwise(pl.lit(0))
                        .alias("ppi_interaction")
                    )

                    subset_data = subset_data.sort("graph_index").drop(["graph_index", "Data Partition"])

                    ppi_interaction_count = (subset_data["ppi_interaction"] == 1).sum()
                    ppi_interaction_percent = (ppi_interaction_count / subset_data.shape[0]) * 100

                    rbp1_binding_count = (subset_data[binding_col1] == 1).sum()
                    rbp1_binding_percent = (rbp1_binding_count / subset_data.shape[0]) * 100

                    rbp2_binding_count = (subset_data[binding_col2] == 1).sum()
                    rbp2_binding_percent = (rbp2_binding_count / subset_data.shape[0]) * 100

                    # Prepare the data for OLS regression
                    subset_data = subset_data.to_pandas()
                    y = subset_data["target"]

                    X = subset_data.drop(columns=["target"])
                    assert X.shape[1] == 3, logger.error(f"Expected 3 columns in X, but got {X.shape[1]}")

                    # Add a constant to the model (intercept)
                    X = sm.add_constant(X)

                    logger.info(f"Fitting -- {cell_line}: {rbp_pair} @ Pos. {position} | Number of observations: {X.shape[0]}")
                    # Fit the OLS model
                    ols_model = sm.OLS(y, X, missing="raise").fit()

                    # Extract the coefficients and p-values
                    intercept_beta = ols_model.params["const"]
                    rbp1_beta = ols_model.params[binding_col1]
                    rbp2_beta = ols_model.params[binding_col2]
                    interaction_beta = ols_model.params["ppi_interaction"]

                    intercept_p_value = ols_model.pvalues["const"]
                    rbp1_p_value = ols_model.pvalues[binding_col1]
                    rbp2_p_value = ols_model.pvalues[binding_col2]
                    interaction_p_value = ols_model.pvalues["ppi_interaction"]

                    # Append the results to a list
                    results.append(
                        {
                            "Data Partition": "All",
                            "Cell Line": cell_line,
                            "RBP Pair": rbp_pair,
                            "Position": position,
                            "# Graphs Used in OLS Reg.": X.shape[0],
                            "# Rows - RBP 1 Bound": rbp1_binding_count,
                            "% Rows - RBP 1 Bound": rbp1_binding_percent,
                            "# Rows - RBP 2 Bound": rbp2_binding_count,
                            "% Rows - RBP 2 Bound": rbp2_binding_percent,
                            "# Rows - PPI Interaction": ppi_interaction_count,
                            "% Rows - PPI Interaction": ppi_interaction_percent,
                            "Intercept Beta Coefficient": intercept_beta,
                            "Intercept Param P-value": intercept_p_value,
                            "RBP 1 Beta Coefficient": rbp1_beta,
                            "RBP 1 Param P-value": rbp1_p_value,
                            "RBP 2 Beta Coefficient": rbp2_beta,
                            "RBP 2 Param P-value": rbp2_p_value,
                            "Interaction Beta Coefficient": interaction_beta,
                            "Interaction Param P-value": interaction_p_value,
                        }
                    )

            # Convert the results to a DataFrame and save to a file
            results_df = pd.DataFrame(results).sort_values("Interaction Param P-value")

            for cell_line in self.cell_lines:
                num_graphs = results_df[results_df["Cell Line"] == cell_line]["# Graphs Used in OLS Reg."].unique()
                assert len(num_graphs) == 1, f"Number of graphs used in OLS regression is not consistent for {cell_line}."

            results_df.to_csv(OLS_LIN_REG_CACHE_FILE, sep="\t", index=False)

            self.ppi_ols_lin_reg_results = results_df
            logger.success("Pair-position combo OLS linear regression results saved.")
            
    
    def run_all_ppi_features_and_interaction_term_ols_linear_regression(self): 
            
        OLS_LIN_REG_CACHE_FILE="../output/ols_lin_reg_ppi_results/all_ppi_features_and_interaction_terms_ols_lin_reg_results.tsv"

        if pathlib.Path(OLS_LIN_REG_CACHE_FILE).exists():
            self.all_ppi_features_ols_lin_reg_results = pd.read_csv(OLS_LIN_REG_CACHE_FILE, sep="\t")
            logger.success("FROM CACHE: loaded OLS linear regression results.")
            return self.all_ppi_features_ols_lin_reg_results.head()
        

        else: 
            logger.info("NO CACHE... hence, running OLS linear regression for all PPI features and interaction terms.")

            if not hasattr(self, 'ppi_ols_lin_reg_results'):
                self.run_pair_position_ols_linear_regression()

            unique_combinations = {}
            for cell_line in self.cell_lines:
                cell_line_data = self.ppi_ols_lin_reg_results[self.ppi_ols_lin_reg_results["Cell Line"] == cell_line]
                unique_combinations[cell_line] = sorted(list(cell_line_data[["RBP Pair", "Position"]].drop_duplicates().itertuples(index=False, name=None)))

            results = []
            for cell_line in self.cell_lines:
                original_data = self.xgboost_ppi[cell_line]
                unique_graph_ids = sorted(original_data["graph_index"].unique().to_list())

                regression_data = None

                for rbp_pair, position in unique_combinations[cell_line]:
                    rbp1, rbp2 = rbp_pair.split("-")
                    binding_col1 = f"{rbp1}_{self.position_inverted_dict[str(position)]}"
                    binding_col2 = f"{rbp2}_{self.position_inverted_dict[str(position)]}"
                    interaction_col = f"{rbp1}-{rbp2}_{position}_interaction"

                    subset_data = original_data.select(
                        ["target", "graph_index", binding_col1, binding_col2]
                    ).unique().sort("graph_index")

                    # Rename the binding columns using the splice junction renaming dictionary
                    new_binding_col1 = f"{rbp1}_{position}"
                    new_binding_col2 = f"{rbp2}_{position}"
                    subset_data = subset_data.rename({binding_col1: new_binding_col1, binding_col2: new_binding_col2})

                    subset_data = subset_data.with_columns(
                        pl.when((pl.col(new_binding_col1) == 1) & (pl.col(new_binding_col2) == 1))
                        .then(pl.lit(1))
                        .otherwise(pl.lit(0))
                        .alias(interaction_col)
                    )

                    if regression_data is None:
                        regression_data = subset_data

                    else:
                        assert regression_data["graph_index"].to_list() == subset_data["graph_index"].to_list(), logger.error(f"Mismatch in 'graph_index' columns between regression_data and subset_data for {cell_line} - {rbp_pair} - {position}.")
                        assert interaction_col not in regression_data.columns, logger.error(f"Column '{interaction_col}' already exists in regression_data for {cell_line} - {rbp_pair} - {position}.")
                        
                        subset_data = subset_data.drop("target")

                        duplicate_columns = [col for col in [new_binding_col1, new_binding_col2, interaction_col] if col in regression_data.columns]
                        if len(duplicate_columns) > 0:
                            logger.info(f"Dropping duplicate columns: {duplicate_columns}")
                            subset_data = subset_data.drop(duplicate_columns)
                        
                        regression_data = regression_data.join(subset_data, on="graph_index", how="inner", validate="1:1")

                regression_data = regression_data.sort("graph_index")
                assert regression_data["graph_index"].to_list() == unique_graph_ids, logger.error(f"Mismatch in 'graph_index' columns between regression_data and unique_graph_ids for {cell_line}.")
                
                regression_data = regression_data.drop("graph_index").to_pandas()
                
                expected_columns = list(
                    set(
                        ["target"] + 
                        [f"{rbp}_{position}" for rbp_pair, position in unique_combinations[cell_line] for rbp in rbp_pair.split("-")] + 
                        [f"{rbp_pair.split('-')[0]}-{rbp_pair.split('-')[1]}_{position}_interaction" for rbp_pair, position in unique_combinations[cell_line]]
                    )
                )
                assert sorted(regression_data.columns) == sorted(expected_columns), logger.error(f"Expected columns do not match the actual columns for {cell_line}.")

                logger.info(f"Created input dataset for {cell_line} OLS linear regression model... fitting the model now.")

                y = regression_data["target"]
                X = regression_data.drop(columns=["target"])

                X = sm.add_constant(X)
                ols_model = sm.OLS(y, X, missing="raise").fit()

                results_dict = {
                        "Data Partition": "All",
                        "Cell Line": cell_line,
                        "# Unique Graphs": X.shape[0],
                        "# Features": X.shape[1],
                        "R2 Score": ols_model.rsquared,
                        "Adj. R2 Score": ols_model.rsquared_adj,
                    }
                
                # Ensure params and pvalues have the same indices
                assert ols_model.params.index.equals(ols_model.pvalues.index), "Params and pvalues indices do not match."
                # Sort params and pvalues indices
                sorted_params = ols_model.params.sort_index()
                sorted_pvalues = ols_model.pvalues.sort_index()

                # Convert sorted params and pvalues to 1D arrays with formatted indices
                sorted_params_dict = {f"{index}_beta": value for index, value in sorted_params.items()}
                sorted_pvalues_dict = {f"{index}_raw_pval": value for index, value in sorted_pvalues.items()}

                results_dict.update(sorted_params_dict)
                results_dict.update(sorted_pvalues_dict)

                results.append(results_dict)
                logger.success(f"Finished fitting OLS linear regression model for {cell_line}.")
            
            results_df = pd.DataFrame(results)

            non_feature_columns = [col for col in results_df.columns if "_" not in col]
            feature_columns = sorted([col for col in results_df.columns if "_" in col])
            results_df = results_df[non_feature_columns + feature_columns]

            results_df.to_csv(OLS_LIN_REG_CACHE_FILE, sep="\t", index=False)

            self.all_ppi_features_ols_lin_reg_results = results_df
            logger.success("OLS linear regression results for all PPI features and interaction terms saved.")


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

