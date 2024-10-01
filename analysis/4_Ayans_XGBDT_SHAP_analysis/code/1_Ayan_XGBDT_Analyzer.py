from dataclasses import dataclass
from loguru import logger 

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

import os, json, glob, scipy, concurrent.futures, tqdm, gc, pathlib, pickle

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


    def load_rbp_ppi(self):
        
        if pathlib.Path(self.rbp_comparisons_file).exists():
            with open(self.rbp_comparisons_file, "r") as f:
                self.rbp_ppi = json.load(f)
        
            logger.info("FROM CACHE: RBP PPI info loaded")

        else: 

            rbp_ppi = pd.read_excel("../../../inputs/RBP-RBP_PPI/lang_et_al_rec-y2h_screening_results.xlsx")
            rbp_ppi = rbp_ppi[(rbp_ppi["sumIS"]>=7.1)]

            for column in ["Protein A", "Protein B", "UniProt accessions A", "UniProt accessions B"]: 
                rbp_ppi[column] = rbp_ppi[column].str.lower()

            uniprot_id_mapping = pd.read_csv("../../../inputs/RBP-RBP_PPI/uniprot_mapping.tsv", sep="\t")

            for column in uniprot_id_mapping: 
                uniprot_id_mapping[column] = uniprot_id_mapping[column].str.lower()

            logger.info("RBP PPI data created & loaded")

    
    def get_rbp_and_position(self, string):

        splitter = string.split("_")
    
        rbp = splitter[0]
        position = self.splice_junction_position_renaming[
            "_".join(splitter[1:3])
        ]

        return rbp, position
    
    
    def _cache_to_featherv2(self, df, file_path):

        if not pathlib.Path(file_path).exists():
            logger.info(f"Caching to feather: {file_path}")
            df.write_ipc(file_path)


    def load_ctrl_only_binding_data(self):
        
        if hasattr(self, 'shap_data'):
            self.delete_SHAP_data()

        if hasattr(self, 'ctrl_only_binding_data'):
            logger.info(f"ALREADY LOADED {self.cell_line} {self.distance_threshold} binding data")
            return self.ctrl_only_binding_data.head()

        else:
            cache_file = f"{self.feather_cache}/{self.cell_line}-{self.distance_threshold}-ctrl_only_binding_data.feather"   

            if pathlib.Path(cache_file).exists():
                logger.info(f"LOADING FROM CACHE: {self.cell_line} {self.distance_threshold} CTRL-only binding data loaded")

                self.ctrl_only_binding_data = pl.scan_ipc(cache_file).collect(streaming=True)
                self.binding_columns = [col for col in self.ctrl_only_binding_data.columns if col.endswith("_right") or col.endswith("_left")]

                logger.info(f"{self.cell_line} {self.distance_threshold} binding dataframe shape: {self.ctrl_only_binding_data.shape}")

                return self.ctrl_only_binding_data.head()

            else: 

                logger.info(f"No cache... hence, loading CTRL only binding data for {self.cell_line} {self.distance_threshold}")

                file = glob.glob(
                    f"{self.ayan_binding_folder}/{self.cell_line}-{self.distance_threshold}-*/*.csv.gz"
                )
                assert len(file)==1, logger.error(f"Multiple files found for {self.cell_line} and {self.distance_threshold}: {file}")

                tmp_df = pl.scan_csv(file[0], has_header=True, separator=",",).filter(pl.col("RBP_KD")=="NONE").collect(streaming=True)

                for col in tmp_df.columns: 
                    if col.endswith("_right") or col.endswith("_left"): 
                        tmp_df = tmp_df.with_columns(pl.col(col).cast(pl.Int8))

                self.ctrl_only_binding_data = tmp_df
                self.binding_columns = [col for col in self.ctrl_only_binding_data.columns if col.endswith("_right") or col.endswith("_left")]

                logger.info(f"{self.cell_line} {self.distance_threshold} binding dataframe shape: {self.ctrl_only_binding_data.shape}")

                self._cache_to_featherv2(self.ctrl_only_binding_data, cache_file)

                return self.ctrl_only_binding_data.head()
    

    def plot_upstream_and_downstream_exon_duplication(self): 

        if pathlib.Path(f"../outputs/middle_exon_duplication/{self.cell_line}_inspect_duplication.tsv").exists(): 

            logger.info(f"FROM CACHE: exon duplication CSV file for {self.cell_line} {self.distance_threshold} loaded")

            tmp_df = pd.read_csv(f"../outputs/middle_exon_duplication/{self.cell_line}_inspect_duplication.tsv", sep="\t", index_col = 0)
            
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

        else: 
            
            logger.info(f"Counting exon duplication for {self.cell_line} {self.distance_threshold}")

            ense_counts = {}
            ense_counts["Counts"] = {}
        
            # Collect the unique values from the "ENSE" column
            unique_ense_values = tmp_df.select(pl.col("ENSE").unique()).to_series()
            
            # Iterate through each unique value in the "ENSE" column
            for ense_value in unique_ense_values:
                
                # Count occurrences of the unique value in the "ENSE_UP" column
                count = (tmp_df.filter(pl.col("ENSE_UP") == ense_value).height) + (tmp_df.filter(pl.col("ENSE_DN") == ense_value).height)
                # Store the count in the dictionary
                ense_counts["Counts"][ense_value] = count
        
            tmp_df = pd.DataFrame.from_dict(ense_counts)
        
            tmp_df = tmp_df.sort_values("Counts", ascending=False)
            tmp_df.to_csv(f"../outputs/middle_exon_duplication/{self.cell_line}_inspect_duplication.tsv", sep="\t")

            logger.info(f"Finished counting exon duplication for {self.cell_line} {self.distance_threshold}")

            return tmp_df 


    def partition_dataframe_by_PSI(self, df = None, column_name=None, psi_cutoffs=None): 

        logger.info(psi_cutoffs)

        return {
                    f"PSI < {psi_cutoffs[0]}": 
                        df.filter(pl.col(column_name) < psi_cutoffs[0]),

                    f"PSI >= {psi_cutoffs[0]} & <= {psi_cutoffs[1]}": 
                        df.filter((pl.col(column_name) >= psi_cutoffs[0]) & (pl.col(column_name) <= psi_cutoffs[1])),
                        
                    f"PSI > {psi_cutoffs[1]}":
                        df.filter(pl.col(column_name) > psi_cutoffs[1])
                }
    

    def run_parallel_chi_square_tests(self):

        output_file = f"../outputs/chi_square/feature_specific/tables/{self.cell_line}_chi_square_results.tsv"

        if pathlib.Path(output_file).exists():
            logger.info(f"FROM CACHE: chi-square results for {self.cell_line} loaded")

            self.feature_chi_square_results = pd.read_csv(output_file, sep="\t")
            return self.feature_chi_square_results

        else:

            logger.info(f"Running parallel chi-square tests for {self.cell_line} {self.distance_threshold}")

            feature_chi_square_results = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self.run_chi_square_test, column=column) for column in self.binding_columns]

                with tqdm.tqdm(total=len(futures)) as pbar:

                    for future in concurrent.futures.as_completed(futures):
                        feature_chi_square_results.append(future.result())
                        pbar.update(1)
            
            self.feature_chi_square_results = pd.concat(feature_chi_square_results).sort_values("Statistic", ascending=False)

            self.feature_chi_square_results.to_csv(output_file, index=False, sep="\t")

            logger.info(f"Finished parallel chi-square tests for {self.cell_line} {self.distance_threshold}")
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

        lists = pd.DataFrame.from_dict(results, orient="index").to_numpy().tolist()
        
        try: 
            result = scipy.stats.chi2_contingency(lists)
            return pd.DataFrame(
                [[column, result.statistic, result.pvalue]], 
                columns=["Feature", "Statistic", "P-Val"]
            )
        
        except ValueError: 
            return pd.DataFrame(
                [[column, None, None]], 
                columns = ["Feature", "Statistic", "P-Val"]
            ) 
        
    
    def convert_features_to_rbp_position_matrix(self, column=None):
        
        heatmap_df = {}

        for index, data in self.feature_chi_square_results.iterrows(): 
            rbp, position = self.get_rbp_and_position(data["Feature"])
            
            if rbp not in heatmap_df: 
                heatmap_df[rbp] = {}

            heatmap_df[rbp][position] = data[column]

        return pd.DataFrame.from_dict(heatmap_df, orient="columns").sort_index(axis=0).sort_index(axis=1)
    

    def plot_feature_chi_square_results(self, column=None): 

        heatmap_df = self.convert_features_to_rbp_position_matrix(column=column)

        plt.figure(dpi=200,figsize=(30,10))
        
        _=plt.hist(heatmap_df.to_numpy().flatten(), bins=200)

        plt.title(f"{self.cell_line} CTRL ONLY: Per-Feature Chi-Square {column} Histogram", pad=10, fontsize=20)
        plt.show()

        plt.figure(dpi=200, figsize=(50,10))

        mask = heatmap_df.isnull()
        
        cmap = sns.color_palette("Blues", as_cmap=True)
        cmap.set_bad("salmon")

        vmax = 10000
        sns.heatmap(heatmap_df, cmap=cmap, mask=mask, vmax=vmax)

        plt.title(f"{self.cell_line} CTRL ONLY: Per-Feature Chi-Square Statistic Heatmap (Capped at {vmax})", pad=40, fontsize=40)
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


    def load_SHAP_data(self): 

        if hasattr(self, 'ctrl_only_binding_data'):
            self.delete_binding_data()
        
        if hasattr(self, 'shap_data'):
            logger.info(f"ALREADY LOADED {self.cell_line} {self.distance_threshold} SHAP data")
            return self.shap_data.head()

        else:
            cache_file = f"{self.feather_cache}/{self.cell_line}-{self.distance_threshold}-shap_data.feather"

            if pathlib.Path(cache_file).exists():
                logger.info(f"LOADING FROM CACHE: {self.cell_line} {self.distance_threshold} SHAP data loaded")

                self.shap_data = pl.scan_ipc(cache_file).collect(streaming=True)
                logger.info(f"{self.cell_line} {self.distance_threshold} SHAP dataframe shape: {self.shap_data.shape}")

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
                    
                    tmp_df = pl.scan_csv(file, has_header=True, separator=",").collect(streaming=True)
                    tmp_df = tmp_df.with_columns(
                        pl.lit(file.split("-")[-2]).alias("Data Partition")
                    )

                    dataframes.append(tmp_df)

                tmp_df = pl.concat(dataframes, how="diagonal")

                for col in tmp_df.columns: 
                    if col.endswith("_right") or col.endswith("_left"): 
                        tmp_df = tmp_df.with_columns(pl.col(col).cast(pl.Int8))
                
                self.shap_data = tmp_df

                self.binding_columns = [col for col in self.shap_data.columns if col.endswith("_right") or col.endswith("_left")]
                self.shap_columns = [col for col in self.shap_data.columns if col.endswith("_shap")]

                self._cache_to_featherv2(self.shap_data, cache_file)

                logger.info(f"{self.cell_line} {self.distance_threshold} SHAP dataframe shape: {self.shap_data.shape}")
                return self.shap_data.head()


    def predicted_vs_actual_PSI_model(self): 

        plt.figure(dpi=200, figsize=(10,10))
        
        joint_plot = sns.jointplot(
            data = self.shap_data,  
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