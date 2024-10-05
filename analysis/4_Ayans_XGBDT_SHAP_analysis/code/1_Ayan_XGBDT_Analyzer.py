import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from dataclasses import dataclass
from loguru import logger 
from matplotlib.patches import Patch
from statsmodels.stats.multitest import multipletests

import os, json, glob, scipy, concurrent.futures, tqdm, gc, pathlib, pickle, argparse, sys

@dataclass
class AyanXgbdtAnalyzer:
    # initate the class with the following parameters
    cell_line: str = None  
    distance_threshold: int = None

    ##########################################
    # General (non-class specific) variables #
    ##########################################

    ayan_shap_folder = "/project/PlatigLab/data/collaborators/BWH/3_XGBDT_SHAP_data_2024_07/"
    ayan_binding_folder = "/project/PlatigLab/data/collaborators/BWH/2_input_binding_data_and_INCORRECT_SHAP_toy_data_2024-07/input_binding_data/"

    feather_cache = "../outputs/__featherv2-cache__"

    shap_actual_psi_column = "target"
    shap_predicted_psi_column = "psi_hat"
    binding_psi_column = "psi"

    # RBP PPI 
    rbp_comparisons_file = "../outputs/rbp_comparisons/rbp_comparisons.json"

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

    #############
    # Functions #
    #############


    def __post_init__(self):

        self.load_rbp_ppi()

        logger.add(sys.stdout)


    def load_rbp_ppi(self):
        
        if pathlib.Path(self.rbp_comparisons_file).exists():
            with open(self.rbp_comparisons_file, "r") as f:
                self.rbp_ppi = json.load(f)
        
            logger.info("FROM CACHE: RBP PPI info loaded")

        # else: 

        #     rbp_ppi = pd.read_excel("../../../inputs/RBP-RBP_PPI/lang_et_al_rec-y2h_screening_results.xlsx")
        #     rbp_ppi = rbp_ppi[(rbp_ppi["sumIS"]>=7.1)]

        #     for column in ["Protein A", "Protein B", "UniProt accessions A", "UniProt accessions B"]: 
        #         rbp_ppi[column] = rbp_ppi[column].str.lower()

        #     uniprot_id_mapping = pd.read_csv("../../../inputs/RBP-RBP_PPI/uniprot_mapping.tsv", sep="\t")

        #     for column in uniprot_id_mapping: 
        #         uniprot_id_mapping[column] = uniprot_id_mapping[column].str.lower()

        #     logger.success("RBP PPI data created & loaded")

    
    def get_rbp_and_position(self, string):

        splitter = string.split("_")
    
        rbp = splitter[0]
        position = self.splice_junction_position_renaming[
            "_".join(splitter[1:3])
        ]

        return rbp, position
    
    
    def _cache_to_featherv2(self, df, file_path):

        if not pathlib.Path(file_path).exists():
            logger.info(f"Caching to Feather V2 (Arrow): {file_path}")

            df.write_ipc(file_path, compression="lz4")
            logger.success(f"Finished creating Feather V2 (Arrow) file for {file_path}")



    def load_ctrl_only_binding_data(self, column=None):
        
        if hasattr(self, 'shap_data'):
            self.delete_SHAP_data()

        if hasattr(self, 'ctrl_only_binding_data'):
            logger.info(f"ALREADY LOADED {self.cell_line} {self.distance_threshold} binding data")
            return self.ctrl_only_binding_data.head()

        else:
            cache_file = f"{self.feather_cache}/{self.cell_line}-{self.distance_threshold}-ctrl_only_binding_data.feather"   

            if pathlib.Path(cache_file).exists():
                logger.info(f"FROM CACHE: {self.cell_line} {self.distance_threshold} CTRL-only binding data loaded")

                ctrl_only_binding_data = pl.scan_ipc(cache_file).collect(streaming=True)

                self.binding_columns = [col for col in ctrl_only_binding_data.columns if col.endswith("_right") or col.endswith("_left")]
                ctrl_only_binding_data = ctrl_only_binding_data.with_columns([pl.col(col).cast(pl.Int8) for col in self.binding_columns])

                self.ctrl_only_binding_data = ctrl_only_binding_data

                logger.info(f"{self.cell_line} {self.distance_threshold} binding dataframe shape: {self.ctrl_only_binding_data.shape}")
                return self.ctrl_only_binding_data.head()

            else: 

                logger.info(f"No cache... hence, loading CTRL only binding data for {self.cell_line} {self.distance_threshold}")

                file = glob.glob(
                    f"{self.ayan_binding_folder}/{self.cell_line}-{self.distance_threshold}-*/*.csv.gz"
                )
                assert len(file)==1, logger.error(f"Multiple files found for {self.cell_line} and {self.distance_threshold}: {file}")

                tmp_df = pl.scan_csv(file[0], has_header=True, separator=",",)

                if column is not None:
                    tmp_df = tmp_df.select(column, "psi", "RBP_KD")
                
                tmp_df = tmp_df.filter(pl.col("RBP_KD")=="NONE").unique().collect(streaming=True)
                self.binding_columns = [col for col in tmp_df.columns if col.endswith("_right") or col.endswith("_left")]

                if len(self.binding_columns) > 0:
                    tmp_df = tmp_df.with_columns([pl.col(col).cast(pl.Int8) for col in self.binding_columns])

                self.ctrl_only_binding_data = tmp_df
                logger.success(f"{self.cell_line} {self.distance_threshold} binding dataframe loaded. Shape: {self.ctrl_only_binding_data.shape}")

                self._cache_to_featherv2(self.ctrl_only_binding_data, cache_file)

                return self.ctrl_only_binding_data.head()
    

    def plot_upstream_and_downstream_exon_duplication(self): 

        if pathlib.Path(f"../outputs/middle_exon_duplication/{self.cell_line}_inspect_duplication.csv").exists(): 

            logger.info(f"FROM CACHE: exon duplication CSV file for {self.cell_line} {self.distance_threshold} loaded")

            tmp_df = pd.read_csv(f"../outputs/middle_exon_duplication/{self.cell_line}_inspect_duplication.csv", sep=",", index_col = 0)
            
            fig, axes = plt.subplots(1, 2, figsize=(12,3))
            
            # Histogram on the left subplot
            axes[0].hist(tmp_df['Counts'], bins=200, color='blue', edgecolor='black')
            axes[0].set_title('Histogram of Counts')
            axes[0].set_xlabel('Counts')
            axes[0].set_ylabel('Frequency')
            
            # Boxplot on the right subplot
            axes[1].boxplot(tmp_df['Counts'], vert=False)
            axes[1].set_title('Boxplot of Counts')
            axes[1].set_xlabel('Counts')

            plt.suptitle(f"{self.cell_line}-{self.distance_threshold}: Total Times 'Middle Exon' Seen \nin Upstream + Downstream Combined", fontsize=20,)
            
            plt.tight_layout()
            plt.show()
                
            return tmp_df

        # else: 
            
        #     logger.info(f"Counting exon duplication for {self.cell_line} {self.distance_threshold}")

        #     ense_counts = {}
        #     ense_counts["Counts"] = {}
        
        #     # Collect the unique values from the "ENSE" column
        #     unique_ense_values = tmp_df.select(pl.col("ENSE").unique()).to_series()
            
        #     # Iterate through each unique value in the "ENSE" column
        #     for ense_value in unique_ense_values:
                
        #         # Count occurrences of the unique value in the "ENSE_UP" column
        #         count = (tmp_df.filter(pl.col("ENSE_UP") == ense_value).height) + (tmp_df.filter(pl.col("ENSE_DN") == ense_value).height)
        #         # Store the count in the dictionary
        #         ense_counts["Counts"][ense_value] = count
        
        #     tmp_df = pd.DataFrame.from_dict(ense_counts)
        
        #     tmp_df = tmp_df.sort_values("Counts", ascending=False)
        #     tmp_df.to_csv(f"../outputs/middle_exon_duplication/{self.cell_line}_inspect_duplication.tsv", sep="\t")

        #     logger.success(f"Finished counting exon duplication for {self.cell_line} {self.distance_threshold}")

        #     return tmp_df 


    def get_number_SLURM_CPUs(self):
        slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")

        if slurm_cpus is not None:
            return int(slurm_cpus)
        else:
            logger.warning("SLURM_CPUS_PER_TASK environment variable not set. Defaulting to 1 CPU.")
            return 1



    def partition_dataframe_by_PSI(self, df = None, column_name=None, psi_cutoffs=None): 

        return {
                    f"PSI < {psi_cutoffs[0]}": 
                        df.filter(pl.col(column_name) < psi_cutoffs[0]),

                    f"PSI >= {psi_cutoffs[0]} & <= {psi_cutoffs[1]}": 
                        df.filter((pl.col(column_name) >= psi_cutoffs[0]) & (pl.col(column_name) <= psi_cutoffs[1])),
                        
                    f"PSI > {psi_cutoffs[1]}":
                        df.filter(pl.col(column_name) > psi_cutoffs[1])
                }
    

    def run_parallel_chi_square_tests(self):

        chi_square_output = f"../outputs/chi_square/feature_specific/tables/{self.cell_line}_chi_square_results.tsv"
        chi_square_contingency_output = f"../outputs/chi_square/feature_specific/tables/{self.cell_line}_chi_square_contingency_tables.pkl"

        if pathlib.Path(chi_square_output).exists() and pathlib.Path(chi_square_contingency_output).exists():
            logger.info(f"FROM CACHE: chi-square results for {self.cell_line} loaded")

            self.feature_chi_square_results = pd.read_csv(chi_square_output, sep="\t")
            self.feature_chi_square_contingency_tables = pickle.load(open(chi_square_contingency_output, "rb"))

            return self.feature_chi_square_results
        
        else:

            logger.info(f"Running parallel chi-square tests for {self.cell_line} {self.distance_threshold}")

            feature_chi_square_results = []
            feature_chi_square_contingency_tables = {}

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.get_number_SLURM_CPUs()) as executor:
                futures = {executor.submit(self.run_chi_square_test, column=column): column for column in self.binding_columns}

                with tqdm.tqdm(total=len(futures)) as pbar:

                    for future in concurrent.futures.as_completed(futures):
                        column = futures[future]
                        function_results = future.result()

                        feature_chi_square_results.append(function_results["Chi-Square Test"])
                        feature_chi_square_contingency_tables[column] = function_results["Contingency Table"]

                        pbar.update(1)
            
            self.feature_chi_square_results = pd.concat(feature_chi_square_results).sort_values("Statistic", ascending=False)
            self.feature_chi_square_results.to_csv(chi_square_output, index=False, sep="\t")

            self.feature_chi_square_contingency_tables = feature_chi_square_contingency_tables
            with open(chi_square_contingency_output, 'wb') as f:
                pickle.dump(self.feature_chi_square_contingency_tables, f)

            logger.success(f"Finished parallel chi-square tests for {self.cell_line} {self.distance_threshold}")
            return self.feature_chi_square_results
    

    def run_chi_square_test(self, column = None):

        partitions = self.partition_dataframe_by_PSI(df=self.ctrl_only_binding_data, column_name="psi", psi_cutoffs=self.psi_partition_thresholds)

        results = {}
        for key in partitions: 

            results[key] = {}

            count_1 = partitions[key][column].sum()
            count_0 = partitions[key][column].shape[0] - count_1

            results[key]["Bound"] = count_1 
            results[key]["Unbound"] = count_0

        contingency_table = pd.DataFrame.from_dict(results, orient="index")
        
        try: 
            result = scipy.stats.chi2_contingency(contingency_table.to_numpy().tolist())
            return {
                    "Chi-Square Test": pd.DataFrame(
                                            [[column, result.statistic, result.pvalue]], 
                                            columns=["Feature", "Statistic", "P-Val"]
                                        ), 
                    "Contingency Table": contingency_table                
                }
        
        except ValueError: 
            return {
                    "Chi-Square Test": pd.DataFrame(
                                            [[column, None, None]], 
                                            columns=["Feature", "Statistic", "P-Val"]
                                        ), 
                    "Contingency Table": contingency_table                
                }
        
    
    def plot_chi_square_contingency_tables(self): 
        
        plot_df = []

        for key in self.feature_chi_square_contingency_tables: 
            sub_dict = self.feature_chi_square_contingency_tables[key].to_dict(orient="index")

            for psi_threshold in sub_dict:
                plot_df.append(
                    [
                        key,
                        psi_threshold, 
                        (sub_dict[psi_threshold]["Bound"])/(sub_dict[psi_threshold]["Bound"] + sub_dict[psi_threshold]["Unbound"]),
                        self.splice_junction_position_renaming["_".join(key.split("_")[1:3])]
                    ]
                )
        
        plot_df = pd.DataFrame(plot_df, columns=["Feature", "PSI Threshold", "Percent Bound", "Position"])    


        plt.figure(dpi=200, figsize=(15, 5))

        sns.lineplot(data=plot_df, x="PSI Threshold", y="Percent Bound", hue="Position", marker="o")

        plt.title(f"{self.cell_line}: Percent Bound by PSI Threshold and Position", fontsize=20)
        plt.xlabel("PSI Threshold", fontsize=15)
        plt.ylabel("Percent Bound", fontsize=15)
        plt.legend(title="Position", fontsize=12)

        plt.tight_layout()
        plt.show()

        plt.figure(dpi=200, figsize=(15, 5))

        sns.lineplot(data=plot_df, x="PSI Threshold", y="Percent Bound", hue="Feature", marker="o")

        plt.title(f"{self.cell_line}: Percent Bound by PSI Threshold and Position", fontsize=20)
        plt.xlabel("PSI Threshold", fontsize=15)
        plt.ylabel("Percent Bound", fontsize=15)
        plt.legend().set_visible(False)

        plt.tight_layout()
        plt.show()

        return plot_df 



            

    
    def chi_square_convert_features_to_rbp_position_matrix(self, column=None):
        
        heatmap_df = {}

        for index, data in self.feature_chi_square_results.iterrows(): 
            rbp, position = self.get_rbp_and_position(data["Feature"])
            
            if rbp not in heatmap_df: 
                heatmap_df[rbp] = {}

            heatmap_df[rbp][position] = data[column]

        return pd.DataFrame.from_dict(heatmap_df, orient="columns").sort_index(axis=0).sort_index(axis=1)
    

    def plot_feature_chi_square_statistics(self, column=None): 

        heatmap_df = self.chi_square_convert_features_to_rbp_position_matrix(column=column)

        plt.figure(dpi=200,figsize=(30,10))
        
        _=plt.hist(heatmap_df.to_numpy().flatten(), bins=200)

        plt.title(f"{self.cell_line} CTRL ONLY: Per-Feature Chi-Square {column} Histogram", pad=10, fontsize=30)
        plt.xlabel("Chi-Square Statistic", fontsize=20)
        plt.ylabel("Frequency", fontsize=20)
        plt.show()

        plt.figure(dpi=200, figsize=(50,10))

        mask = heatmap_df.isnull()
        cmap = sns.color_palette("Blues", as_cmap=True)
        cmap.set_bad("salmon")

        if column=="Statistic": 
            vmax = 10000
        else: 
            vmax = None

        sns.heatmap(heatmap_df, cmap=cmap, mask=mask, vmax=vmax)

        plt.title(f"{self.cell_line} CTRL ONLY: Per-Feature Chi-Square {column} Heatmap (Capped at {vmax})", pad=40, fontsize=40)
        plt.show()
            
        return heatmap_df


    def delete_binding_data(self): 
        del self.ctrl_only_binding_data
        gc.collect()

        logger.info(f"Deleted {self.cell_line} {self.distance_threshold} binding data")


    def delete_SHAP_data(self):
        del self.shap_data
        gc.collect()

        logger.info(f"Deleted {self.cell_line} {self.distance_threshold} SHAP data")


    def load_SHAP_data(self, column=None): 

        cache_file = f"{self.feather_cache}/{self.cell_line}-{self.distance_threshold}-shap_data.feather"

        if hasattr(self, 'ctrl_only_binding_data'):
            self.delete_binding_data()
        
        if hasattr(self, 'shap_data'):
            logger.info(f"ALREADY LOADED {self.cell_line} {self.distance_threshold} SHAP data")

            if column is not None: 
                self.shap_data = self.shap_data.select(column, "target", "Data Partition")
            
            return self.shap_data.head()

        else:

            if pathlib.Path(cache_file).exists():
                logger.info(f"FROM CACHE: {self.cell_line} {self.distance_threshold} SHAP data loaded")

                shap_data = pl.scan_ipc(cache_file).collect(streaming=True)

                if column is not None: 
                    shap_data = shap_data.select(column, "target", "Data Partition")

                self.shap_columns = [col for col in shap_data.columns if col.endswith("_shap")]
                self.binding_columns = [col for col in shap_data.columns if col.endswith("_right") or col.endswith("_left")]

                shap_data = shap_data.with_columns([pl.col(col).cast(pl.Int8) for col in self.binding_columns])
                self.shap_data = shap_data

                logger.success(f"{self.cell_line} {self.distance_threshold} SHAP dataframe loaded. Shape: {self.shap_data.shape}")
                return self.shap_data.head()
            
            else: 

                logger.info(f"No Cache... hence, loading SHAP data for {self.cell_line} {self.distance_threshold}")

                files = sorted([file for file in glob.glob(f"{self.ayan_shap_folder}/*-{self.cell_line}-{self.distance_threshold}-*/*-data.dat")])
                assert len(files)==3, logger.error([file.split("/")[-1] for file in files])

                column_reference = set(pd.read_csv(files[0], sep=",", nrows=0).columns.to_list())

                dataframes = []

                for file in files: 

                    logger.info(
                        f"{file.split('/')[-1]} is missing the following expected columns: {set(pd.read_csv(file, sep=',', nrows=0).columns.to_list()).symmetric_difference(column_reference)}"
                    )
                    

                    tmp_df = pl.scan_csv(file, has_header=True, separator=",")

                    if column is not None: 
                        tmp_df = tmp_df.select(column, "target",)

                    tmp_df = tmp_df.collect(streaming=True)

                    tmp_df = tmp_df.with_columns(
                        pl.lit(file.split("-")[-2]).alias("Data Partition")
                    )

                    dataframes.append(tmp_df)

                tmp_df = pl.concat(dataframes, how="diagonal")

                self.binding_columns = [col for col in tmp_df.columns if col.endswith("_right") or col.endswith("_left")]
                self.shap_columns = [col for col in tmp_df.columns if col.endswith("_shap")]

                tmp_df = tmp_df.with_columns([pl.col(col).cast(pl.Int8) for col in self.binding_columns])           
                self.shap_data = tmp_df

                self._cache_to_featherv2(self.shap_data, cache_file)

                logger.success(f"{self.cell_line} {self.distance_threshold} SHAP dataframe loaded. Shape: {self.shap_data.shape}")
                return self.shap_data.head()


    def predicted_vs_actual_PSI_model(self): 

        plt.figure(dpi=200, figsize=(10,10))

        plotting_df = self.shap_data.select(["target", "psi_hat"])
        
        joint_plot = sns.jointplot(
            data = plotting_df,  
            x="target", 
            y="psi_hat",
            kind="hist",
            marginal_kws = {"bins": 100}, 
            height=5, 
            stat="percent",
        )

        cax = joint_plot.ax_joint.inset_axes([1.3, 0.1, 0.05, 0.8])
        plt.colorbar(
            joint_plot.ax_joint.collections[0], ax=joint_plot.ax_joint, cax=cax, label="Percentage"
        )

        joint_plot.figure.set_dpi(150)
        plt.suptitle(f"{self.cell_line}: Actual vs Predicted", x=0.5, y=1.0)

        plt.show()
    

    def compare_chromosome_vs_psi(self): 

        tmp_df = self.ctrl_only_binding_data.select(
            ["psi", "psip", "sequence"]
        )

        tmp_df = pl.concat(
            [
                tmp_df.select(["psi", "sequence"]),
                tmp_df.select(["psip", "sequence"]).rename({"psip": "psi"})
            ]
        )

        tmp_df = tmp_df.filter(pl.col("psi")<=1.01)

        unique_sequences = sorted(tmp_df["sequence"].unique())

        fig, axs = plt.subplots(4, 6, sharex=True, sharey=True, figsize=(30,10), dpi=200)
        
        # Flatten the axs array for easier indexing
        axs = axs.flatten()
        
        # Plot histograms for each unique sequence
        for i, sequence in enumerate(unique_sequences):
            subset = tmp_df.filter(pl.col("sequence") == sequence)
            _=axs[i].hist(subset["psi"], bins=100)
            _=axs[i].set_title(f'{sequence}')

        fig.suptitle(f"{self.cell_line}: PSI Histograms by Chromosome", fontsize=40)
        fig.supxlabel("PSI", fontsize=30)
        fig.supylabel("Frequency", fontsize=30, x=-0.02)
        
        plt.tight_layout()
        plt.show()


    def run_parallel_feature_specific_kruskal_wallis(self): 

        output_file = f"../outputs/kruskal_wallis/feature_specific/{self.cell_line}_kruskal_wallis.tsv"

        if pathlib.Path(output_file).exists():
            logger.info(f"FROM CACHE: Kruskal-Wallis results for {self.cell_line} {self.distance_threshold} loaded")
            self.feature_specific_kruskal_wallis = pd.read_csv(output_file, sep="\t")
            
            return self.feature_specific_kruskal_wallis.head()
        
        else: 

            logger.info(f"Running Kruskal-Wallis tests for {self.cell_line} {self.distance_threshold}")

            partitions = self.partition_dataframe_by_PSI(
                df = self.shap_data, 
                column_name = "target", 
                psi_cutoffs=self.psi_partition_thresholds
            )

            kruskal_df = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.get_number_SLURM_CPUs()) as executor:
                futures = {
                    executor.submit(self.run_kruskal_wallis_test, col, partitions): col for col in self.shap_columns
                }

                for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Running Kruskal-Wallis tests"):
                    kruskal_df.append(future.result())

            self.feature_specific_kruskal_wallis = pd.DataFrame(kruskal_df, columns=["Feature", "Statistic", "P-Val"]).sort_values("Statistic", ascending=False)
            self.feature_specific_kruskal_wallis.to_csv(output_file, sep="\t", index=False)
            logger.success(f"{self.cell_line} {self.distance_threshold} Kruskal-Wallis tests completed and cached.")
            return self.feature_specific_kruskal_wallis.head()

    def run_kruskal_wallis_test(self, col, partitions):

        shap_lists = [partitions[key][col].to_list() for key in partitions]

        kruskal_result = scipy.stats.kruskal(*shap_lists)
        return [col, kruskal_result.statistic, kruskal_result.pvalue]
    

    def parallel_inspect_local_SHAP_by_dataset(self, feature):

        if not hasattr(self, 'shap_data'):
            self.load_SHAP_data(column=feature)

        self.plot_feature_local_SHAP_by_dataset(feature=feature)


    def plot_feature_local_SHAP_by_dataset(self, feature=None): 
                
        assert "_shap" in feature, logger.error(f"Feature {feature} is not a SHAP feature")

        plotting_df = self.shap_data.filter(pl.col(feature) != 0)

        if not plotting_df.is_empty():
            fig, ax = plt.subplots(1,2, dpi=200, figsize=(10,5),)

            sns.histplot(
                data=plotting_df, 
                x=feature, 
                hue="Data Partition", 
                bins=100,
                ax=ax[0] 
            )

            sns.boxplot(
                data=plotting_df, 
                y=feature, 
                x="Data Partition", 
                ax=ax[1] 
            )

            rbp, position = self.get_rbp_and_position(feature)

            plt.suptitle(f"Non-Zero Local SHAP Values by Data Partition\n{self.cell_line} {rbp} {position}: # Rows -- {plotting_df.shape[0]:,}", fontsize=20)

            plt.tight_layout()

            plt.savefig(f"../outputs/local_shap/plots/{self.cell_line}-{feature}.png", bbox_inches="tight"),
            plt.close()

        else: 
            logger.warning(f"{feature} has no non-zero local SHAP values")
    

    def convert_1D_row_to_matrix(self, df=None): 
            
        feature_matrix = {}

        for key, value in df.to_dict(as_series=False).items(): 

            rbp, position = self.get_rbp_and_position(key)
            if rbp not in feature_matrix: 
                feature_matrix[rbp] = {}

            feature_matrix[rbp][position] = value[0]

        return pd.DataFrame(feature_matrix).sort_index(axis=1).sort_index(axis=0)


    def get_global_SHAP_matrix(self, df=None): 

        if not hasattr(self, 'shap_data'):
            self.load_SHAP_data()

        # Take all local SHAP columns and get the absolute value mean (Global SHAP)
        abs_mean_df = df.select(
            [pl.col(col).abs().mean().alias(col) for col in self.shap_columns]
        )

        # plt.figure(dpi=200, figsize=(30,10))

        # sns.histplot(data=abs_mean_df.to_numpy().flatten(), bins=500)

        # plt.title(f"{self.cell_line}: Global SHAP Histogram", fontsize=30)
        # plt.xlabel("Mean Absolute SHAP Value", fontsize=20)
        # plt.ylabel("Frequency", fontsize=20)

        # plt.show()

        return self.convert_1D_row_to_matrix(df = abs_mean_df)


    def get_total_binding_percent(self, df=None): 
    
        if not hasattr(self, 'shap_data'):
            self.load_SHAP_data()

        binding_percent = df.select(
            [(pl.col(col).sum() / pl.count() * 100).alias(col) for col in self.binding_columns]
        )

        return self.convert_1D_row_to_matrix(df = binding_percent)

    
    def get_positional_preference(self, df=None): 
        assert type(df) == pd.DataFrame, logger.error(f"Input must be a pandas DataFrame, not {type(df)}")

        return df.copy(deep=True).apply(lambda x: (x / x.sum())*100)
    

    def plot_global_SHAP_vs_binding_plots(self): 
        
        global_SHAP_matrix = self.get_global_SHAP_matrix(df=self.shap_data)
        total_binding_percent = self.get_total_binding_percent(df=self.shap_data)
        
        nrows=2
        fig, ax = plt.subplots(nrows, 1, figsize=(25,9), dpi=200)

        plt.suptitle(f"{self.cell_line}: Global SHAP vs % Events Bound", fontsize=40, x=0.5, y=1.0)
        
        heatmap = sns.heatmap(global_SHAP_matrix, xticklabels=True, yticklabels=True,ax=ax[0],cmap="Blues", cbar_kws={"pad": 0.01})
        heatmap.collections[0].colorbar.ax.tick_params(labelsize=18)
        ax[0].set_title("Global SHAP", fontsize=24,)

        heatmap = sns.heatmap(total_binding_percent, xticklabels=True, yticklabels=True,ax=ax[1],cmap="Blues", cbar_kws={"pad": 0.01})
        heatmap.collections[0].colorbar.ax.tick_params(labelsize=18)
        ax[1].set_title("% Events Bound", fontsize=24)
    
        plt.tight_layout()
        plt.show()

    
    def plot_partition_global_SHAP_vs_binding_plots(self): 
            
        if not hasattr(self, 'shap_data'):
            self.load_SHAP_data()
        
        partitions = self.partition_dataframe_by_PSI(df=self.shap_data, column_name="target", psi_cutoffs=self.psi_partition_thresholds)

        for key in partitions: 

            global_SHAP_matrix = self.get_global_SHAP_matrix(df=partitions[key])
            total_binding_percent = self.get_total_binding_percent(df=partitions[key])

            nrows=2
            fig, ax = plt.subplots(nrows, 1, figsize=(25,9), dpi=200)

            plt.suptitle(f"{self.cell_line}: {key}", fontsize=40, x=0.5, y=1.0)

            heatmap = sns.heatmap(global_SHAP_matrix, xticklabels=True, yticklabels=True,ax=ax[0],cmap="Blues", cbar_kws={"pad": 0.01})
            heatmap.collections[0].colorbar.ax.tick_params(labelsize=18)
            ax[0].set_title("Global SHAP", fontsize=24,)

            heatmap = sns.heatmap(total_binding_percent, xticklabels=True, yticklabels=True,ax=ax[1],cmap="Blues", cbar_kws={"pad": 0.01})
            heatmap.collections[0].colorbar.ax.tick_params(labelsize=18)
            ax[1].set_title("% Events Bound", fontsize=24)

            plt.tight_layout()
            plt.show()
    
    
    def plot_partition_percent_events_bound(self): 
            
        if not hasattr(self, 'shap_data'):
            self.load_SHAP_data()

        partitions = self.partition_dataframe_by_PSI(df=self.shap_data, column_name="target", psi_cutoffs=self.psi_partition_thresholds)

        fig, axs = plt.subplots(len(partitions), 1, figsize=(30, 12), dpi=200)

        plt.suptitle("Percent Events Bound by Partition", fontsize=40)

        for i, key in enumerate(partitions):
            total_binding_percent = self.get_total_binding_percent(df=partitions[key])
            sns.heatmap(total_binding_percent, xticklabels=True, yticklabels=True, cmap="Blues", cbar_kws={"pad": 0.01}, ax=axs[i])
            axs[i].set_title(f"{self.cell_line}: {key}", fontsize=30)

        plt.tight_layout()
        plt.show()
        
    
    def plot_partition_global_SHAP(self): 
            
        if not hasattr(self, 'shap_data'):
            self.load_SHAP_data()

        partitions = self.partition_dataframe_by_PSI(df=self.shap_data, column_name="target", psi_cutoffs=self.psi_partition_thresholds)
        fig, axs = plt.subplots(len(partitions), 1, figsize=(30, 12), dpi=200)

        plt.suptitle("Global SHAP by Partition", fontsize=40)

        for i, key in enumerate(partitions): 
            global_SHAP_matrix = self.get_global_SHAP_matrix(df=partitions[key])
            sns.heatmap(global_SHAP_matrix, xticklabels=True, yticklabels=True, cmap="Blues", cbar_kws={"pad": 0.01}, ax=axs[i])
            axs[i].set_title(f"{self.cell_line}: {key}", fontsize=30)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Ayan XGBDT Analyzer")
    parser.add_argument("--cell_line", type=str, required=True, help="Cell line to analyze")
    parser.add_argument("--distance", type=int, required=True, help="Distance threshold")
    parser.add_argument("--parallel-task", type=str, required=True, help="Which analysis to run")
    parser.add_argument("--feature", type=str, required=True, help="Feature to analyze",)

    args = parser.parse_args()

    analyzer = AyanXgbdtAnalyzer(cell_line=args.cell_line, distance_threshold=args.distance)

    match args.parallel_task:
        case "local_shap_inspection": 
            analyzer.parallel_inspect_local_SHAP_by_dataset(args.feature) 
        case _:
            logger.error(f"Unknown parallel task: {args.parallel_task}")