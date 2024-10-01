from dataclasses import dataclass
from loguru import logger 

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

import os, json, glob, scipy, concurrent.futures, tqdm, gc



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
        self.load_ctrl_only_binding_data()


    def load_rbp_ppi(self):
        
        if os.path.exists(self.rbp_comparisons_file):
            with open(self.rbp_comparisons_file, "r") as f:
                self.rbp_ppi = json.load(f)
        
            logger.info("Loaded previously created RBP PPI info")

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
    

    def load_ctrl_only_binding_data(self):
        
        logger.info(f"LOADING... CTRL only binding data for {self.cell_line} {self.distance_threshold}")

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
        
        logger.info(f"{self.cell_line} {self.distance_threshold} Dataframe shape: {self.ctrl_only_binding_data.shape}")
    

    def plot_upstream_and_downstream_exon_duplication(self): 

        if os.path.exists(f"../outputs/middle_exon_duplication/{self.cell_line}_inspect_duplication.csv"): 

            logger.info(f"Loading previously created exon duplication CSV file for {self.cell_line} {self.distance_threshold}")

            tmp_df = pd.read_csv(f"../outputs/middle_exon_duplication/{self.cell_line}_inspect_duplication.csv", index_col = 0)
            
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
            tmp_df.to_csv(f"../outputs/middle_exon_duplication/{self.cell_line}_inspect_duplication.csv")

            return tmp_df 


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

        logger.info(f"Running parallel chi-square tests for {self.cell_line} {self.distance_threshold}")

        feature_chi_square_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(self.run_chi_square_test, column=column) for column in self.binding_columns]

            with tqdm.tqdm(total=len(futures)) as pbar:

                for future in concurrent.futures.as_completed(futures):
                    feature_chi_square_results.append(future.result())
                    pbar.update(1)
        
        self.feature_chi_square_results = pd.concat(feature_chi_square_results).sort_values("Statistic", ascending=False)

        logger.info(f"Finished parallel chi-square tests for {self.cell_line} {self.distance_threshold}")
        return self.feature_chi_square_results
    

    def run_chi_square_test(self, column = None):

        partitions = self.partition_dataframe_by_PSI(df=self.ctrl_only_binding_data, column_name=column, psi_cutoffs=self.psi_partition_thresholds)

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

        plt.title(f"{self.cell_line} CTRL ONLY: Per-Feature Chi-Square Statistic Histogram", pad=10, fontsize=20)
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


    