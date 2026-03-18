import yaml, pathlib, glob, json, argparse, os, sys, copy, pickle, copy, gzip

import pandas as pd, polars as pl, seaborn as sns, numpy as np, matplotlib.pyplot as plt, matplotlib.patches as mpatches
import networkx as nx, adjustText

from dataclasses import dataclass
from loguru import logger
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from tqdm import tqdm
from itertools import combinations
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, ttest_ind
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score
from statannotations.Annotator import Annotator
from matplotlib.gridspec import GridSpec

@dataclass
class SecondOrderShapNetworkAnalyzer:

    YAML_CONFIG_FILE = "./1_variable_config.yaml"


#########################################################
############ INITIALIZATION FUNCTIONS ###################
#########################################################


    def __post_init__(self):
        
        with open(self.YAML_CONFIG_FILE, "r") as file:
            self.CONFIG = yaml.safe_load(file)

        self.load_features()
        self.load_ppi()

        logger.success("Class initialized and loaded.")


    def load_features(self):

        rbp_feature_metadata = {}
        for cell_line in self.CONFIG["CELL_LINES"]:
            rbp_feature_metadata[cell_line] = {}

            files = glob.glob(
                f"{self.CONFIG['INTERACTION_VALUES_DIR']}/{cell_line}_*.feather"
            )

            columns = pl.scan_ipc(files[0]).collect_schema().names()
            shap_cols = [col for col in columns if col.endswith('-shap')]
            assert len(shap_cols) == len(columns) - 1, "Expected all columns except index to be SHAP value columns."

            rbp_feature_metadata[cell_line]["Features"] = shap_cols

            rbp_feature_metadata[cell_line]["RBPs"] = sorted(
                list(
                    {
                        col.split("_")[0] for col in shap_cols if col.endswith("-main-shap")
                    }
                )
            )

        output_file = self.CONFIG["RBP_FEATURE_INFO_FILE"]
        with open(output_file, "w") as f:
            json.dump(rbp_feature_metadata, f, indent=4)

        self.rbp_feature_metadata = rbp_feature_metadata


    def load_ppi(self):
        
        OUTPUT_FILE = self.CONFIG["PPI_INFO"]

        # explicitly set dtypes as Rec-Y2H column has so few non-null values that polars infers it as string
        cols = pl.scan_csv(OUTPUT_FILE, separator="\t").collect_schema().names()
        dtypes = {col: pl.String if col in ["Interaction", "RBP 1", "RBP 2"] else pl.Boolean for col in cols}

        self.ppi = pl.read_csv(OUTPUT_FILE, separator="\t", schema_overrides=dtypes)


        # if pathlib.Path(OUTPUT_FILE).exists(): 
        #     logger.success(f"FROM CACHE: loading PPI table from '{OUTPUT_FILE}'...")
        #     self.ppi = pl.read_csv(OUTPUT_FILE, separator="\t")
        #     self.ppi_source_columns = {
        #         "recy2h": "rec-Y2H | Table S2",
        #     }

        # else:
        #     logger.info("Cached PPI file not found. Generating PPI table for both Rec-Y2H and Street et al. Molecular Cell 2024 datasets...")

        #     self.create_protein_synonym_lookup_table(mode="recy2h")
        #     # Read in Rec-Y2H PPI file
        #     recy2h_ppi_table = self.load_rec_y2h_ppi()

        #     final_ppi_table = recy2h_ppi_table
        #     # self.create_protein_synonym_lookup_table(mode="street_et_al")
        #     # street_et_al_ppi_table = self.load_street_et_al_ppi()

        #     # final_ppi_table = pd.merge(
        #     #     recy2h_ppi_table,
        #     #     street_et_al_ppi_table,
        #     #     on=["Cell Line", "Interaction"],
        #     #     how="outer"
        #     # )

        #     final_ppi_table["Interaction"] = final_ppi_table["Interaction"].str.upper()
            
        #     final_ppi_table.sort_values(
        #         by=["Interaction"]
        #     ).to_csv(
        #         OUTPUT_FILE,
        #         sep="\t",
        #         index=False
        #     )
        #     logger.success("SUCCESS: PPI table generated and saved.")


    # def create_protein_synonym_lookup_table(self, mode=None):
    #     assert mode in ["recy2h", "street_et_al",], "Mode must be one of 'recy2h' or 'street_et_al'."

    #     logger.info(f"Creating protein synonym lookup table for '{mode}' ...")

    #     # Uniprot mapping file 
    #     uniprot_mapping = self.CONFIG["PPI_INFO"]["recy2h_ppi_dir"]["uniprot_mapping"]
        
    #     if mode == "recy2h":
    #         screen_results_path = self.CONFIG["PPI_INFO"]["recy2h_ppi_dir"]["results"]
    #         all_rbps_screened_path = self.CONFIG["PPI_INFO"]["recy2h_ppi_dir"]["all_rbps_screened"]

    #     elif mode == "street_et_al":
    #         screen_results_path = self.CONFIG["PPI_INFO"]["street_et_al"]["table_s2_full_results"]
    #         all_rbps_screened_path = self.CONFIG["PPI_INFO"]["street_et_al"]["all_rbps_screened"]

    #     # mapping uniprot ids to gene names/synonyms
    #     uniprot_df = pd.read_csv(uniprot_mapping, sep="\t")
    #     uniprot_df["Gene name"] = uniprot_df["Gene name"].str.lower()
    #     uniprot_df["Gene Synonym"] = uniprot_df["Gene Synonym"].str.lower()

    #     if mode == "recy2h":
    #         # load all rec-Y2H screening results
    #         screening_results = pd.read_excel(screen_results_path)
    #         screening_results["Protein A"] = screening_results["Protein A"].str.lower()
    #         screening_results["Protein B"] = screening_results["Protein B"].str.lower()

    #         # Create RBP sets per cell line
    #         rbp_sets = {cell_line: set([rbp.lower() for rbp in self.rbp_feature_metadata[cell_line]["RBPs"]]) for cell_line in self.CONFIG["CELL_LINES"]}

    #         # Concatenate Protein A/B and UniProt accessions A/B, then create mapping
    #         protein_a = screening_results[["Protein A", "UniProt accessions A"]].rename(
    #             columns={"Protein A": "Protein", "UniProt accessions A": "Accession"}
    #         )
    #         protein_b = screening_results[["Protein B", "UniProt accessions B"]].rename(
    #             columns={"Protein B": "Protein", "UniProt accessions B": "Accession"}
    #         )
            
    #         protein_names = pd.concat([protein_a, protein_b], ignore_index=True)
    #         protein_names = protein_names.drop_duplicates(subset=["Protein", "Accession"])
    #         # Assert no duplicate values in the "Protein" column
    #         assert protein_names["Protein"].is_unique, "Duplicate values found in the 'Protein' column."

    #         all_rbps_screened = pd.read_excel(all_rbps_screened_path)
    #         all_rbps_screened = all_rbps_screened[["Gene Symbol"]]
    #         all_rbps_screened["Gene Symbol"] = all_rbps_screened["Gene Symbol"].str.lower()
            
    #         all_rbps_screened = all_rbps_screened.merge(protein_names, left_on="Gene Symbol", right_on="Protein", how="left")

    #         # Assert that there are no cases where only one of 'Protein' or 'Accession' is missing
    #         mask_protein = all_rbps_screened["Protein"].notnull()
    #         mask_accession = all_rbps_screened["Accession"].notnull()
    #         assert ((mask_protein == mask_accession).all()), "Rows found where only one of 'Protein' or 'Accession' is missing."

    #         # Find gene symbols that do not show up in the screening results but were screened
    #         missing_genes = all_rbps_screened[all_rbps_screened["Accession"].isnull()]["Gene Symbol"].tolist()
    #         logger.warning(f"{len(missing_genes)} gene symbols were screened but do not show up in the screening results: {missing_genes}")

    #         # key mapping: protein name -> uniprot accession
    #         protein_to_accession = dict(
    #             zip(protein_names["Protein"], protein_names["Accession"])
    #         )
        
    #     elif mode == "street_et_al":
    #         raise NotImplementedError("Street et al. mode not yet implemented.")
        

    #     # Build lookup table to relate RBP names they mention to the RBP names we have for our eCLIP
    #     protein_synonym_lookup = {cell_line: {} for cell_line in self.CONFIG["CELL_LINES"]}
    #     # synonyms to manually validate later
    #     validate_synonyms = { cell_line: {} for cell_line in self.CONFIG["CELL_LINES"]}

    #     if mode == "recy2h":
    #         for protein, accession in protein_to_accession.items():
    #             for cell_line, rbp_set in rbp_sets.items():

    #                 # Direct match
    #                 if protein in rbp_set:
    #                     protein_synonym_lookup[cell_line][protein] = protein
                        
    #                 else: 
    #                     # Subset uniprot_df for possible synonyms
    #                     subset_df = uniprot_df[
    #                         (uniprot_df["Gene name"] == protein) |
    #                         (uniprot_df["Gene Synonym"] == protein) |
    #                         (uniprot_df["UniProtKB Gene Name ID"] == accession)
    #                     ]
                        
    #                     # no other possible names means that we do not have that RBP eCLIP'd
    #                     if subset_df.empty:
    #                         protein_synonym_lookup[cell_line][protein] = None
                        
    #                     else:
    #                         # Compile all possible names
    #                         all_names = set(
    #                             subset_df["Gene name"].dropna().str.lower().tolist() +
    #                             subset_df["Gene Synonym"].dropna().str.lower().tolist()
    #                         )

    #                         # Find intersection with RBP set
    #                         matches = rbp_set & all_names
    #                         assert len(matches) <= 1, "Expected at most one match per protein."

    #                         # Assign match if found
    #                         if len(matches) == 1:
    #                             validate_synonyms[cell_line][protein] = matches.pop()
    #                             protein_synonym_lookup[cell_line][protein] = None
    #                         elif len(matches) == 0:
    #                             protein_synonym_lookup[cell_line][protein] = None

    #         for cell_line, lookup in protein_synonym_lookup.items():
    #             for protein, synonym in lookup.items():
    #                 if synonym is not None: 
    #                     assert protein ==synonym, f"Mismatch in synonym lookup for protein {protein} in cell line {cell_line}"
                
    #             logger.warning(f"Cell line {cell_line} - validate the following synonyms manually: {validate_synonyms[cell_line]}. NOTE: these synonyms have not been saved. ")
        
    #     elif mode == "street_et_al":
    #         raise NotImplementedError("Street et al. mode not yet implemented.")

    #     self.protein_synonym_lookup = protein_synonym_lookup

    
    # def load_rec_y2h_ppi(self): 
    #     logger.info("Creating rec-Y2H PPI table... ")
    #     logger.warning("REMINDER: Rec-Y2H paper sampled ~98-99% of the entire 2-way RBP interactome but we are assuming that everything is tested and hence, if it is not significantly interacting, it is 'False'.")

    #     # Load all RBPs screened for recy2h
    #     all_rbps_screened_path = self.CONFIG["PPI_INFO"]["recy2h_ppi_dir"]["all_rbps_screened"]
    #     all_rbps_screened = pd.read_excel(all_rbps_screened_path)
    #     all_rbps = all_rbps_screened["Gene Symbol"].str.lower().unique().tolist()

    #     # Generate all possible two-way combinations (sorted, no self-pairs)
    #     rbp_pairs = [tuple(sorted(pair)) for pair in combinations(all_rbps, 2)]

    #     # Load recy2h results and filter by sumIS >= 7.1
    #     recy2h_results = self.CONFIG["PPI_INFO"]["recy2h_ppi_dir"]["results"]
    #     recy2h_df = pd.read_excel(recy2h_results)
    #     recy2h_df = recy2h_df[recy2h_df["sumIS"] >= 7.1]
    #     recy2h_df["Protein A"] = recy2h_df["Protein A"].str.lower()
    #     recy2h_df["Protein B"] = recy2h_df["Protein B"].str.lower()

    #     # Build lookup table: for each RBP, set of RBPs it interacts with
    #     interaction_lookup = {}
    #     for _, row in recy2h_df.iterrows():
    #         a, b = sorted([row["Protein A"], row["Protein B"]])
    #         if a not in interaction_lookup:
    #             interaction_lookup[a] = set()

    #         interaction_lookup[a].add(b)
            
    #         # Also add reverse for completeness
    #         if b not in interaction_lookup:
    #             interaction_lookup[b] = set()
            
    #         interaction_lookup[b].add(a)

    #     # For all pairs, check interaction and eCLIP status per cell line
    #     recy2h_ppi_table = []

    #     for rbp1, rbp2 in rbp_pairs:
    #         # Check if interaction exists in lookup table
    #         interacts = False
    #         if rbp1 in interaction_lookup and rbp2 in interaction_lookup[rbp1]:
    #             interacts = True
    #             assert rbp2 in interaction_lookup and rbp1 in interaction_lookup[rbp2], "Interaction lookup table inconsistent."
    #         elif rbp2 in interaction_lookup and rbp1 in interaction_lookup[rbp2]:
    #             interacts = True
    #             assert rbp1 in interaction_lookup and rbp2 in interaction_lookup[rbp1], "Interaction lookup table inconsistent."

    #         # For each cell line, check if both RBPs have eCLIP
    #         both_eclip = {}
    #         for cell_line in self.CONFIG["CELL_LINES"]:
    #             both_eclip[cell_line] = (rbp1.upper() in self.rbp_feature_metadata[cell_line]["RBPs"]) and (rbp2.upper() in self.rbp_feature_metadata[cell_line]["RBPs"])

    #         interaction = f"{rbp1}-{rbp2}"
    #         row = {
    #             "Interaction": interaction,
    #             "rec-Y2H | Table S2": interacts,
    #         }
    #         # Add columns for each cell line
    #         for cell_line in self.CONFIG["CELL_LINES"]:
    #             row[f"Both eCLIP - {cell_line}"] = both_eclip[cell_line]
    #         recy2h_ppi_table.append(row)

    #     recy2h_ppi_table = pd.DataFrame(recy2h_ppi_table)
    #     return recy2h_ppi_table


    # def load_street_et_al_ppi(self):
    #     # Read in Street et al. Molecular Cell 2024 PPI file
    #     street_ppi_file = self.CONFIG["PPI_INFO"]["street_et_al"]["table_s3_output"]

    #     with open(street_ppi_file, "r") as f:
    #         street_ppi_data = json.load(f)

    #     street_et_al_ppi_table = []
    #     for cell_line, interactions in street_ppi_data.items():
    #         for rbp_pair in interactions:
    #             assert rbp_pair == sorted(rbp_pair), f"RBP pair {rbp_pair} not sorted"
                
    #             # Check if both RBPs are in the feature metadata for this cell line
    #             rbps_list = list(map(str.lower, self.rbp_feature_metadata[cell_line]["RBPs"]))
    #             assert rbp_pair[0] in rbps_list and rbp_pair[1] in rbps_list, f"RBP pair {rbp_pair} not found in feature metadata for cell line {cell_line}"
                
    #             interaction = f"{rbp_pair[0]}-{rbp_pair[1]}"
    #             street_et_al_ppi_table.append({
    #                 "Cell Line": cell_line,
    #                 "Interaction": interaction,
    #                 "Street et al. | Table S3 | IP/MS": True
    #             })

    #     street_et_al_ppi_table = pd.DataFrame(street_et_al_ppi_table)
    #     return street_et_al_ppi_table

#########################################################
################ "UTIL" FUNCTIONS #######################
#########################################################

    def manually_set_num_tasks(self): 
        return 12
    

    def is_interaction_or_main_effect(self, column=None):

        assert column.endswith("-shap"), f"Feature name '{column}' must end with '-shap'."
        
        if column.endswith("-main-shap"): 
            return "main"
        elif column.endswith("-interaction-shap"):
            return "interaction"
        else:
            raise ValueError(f"Feature name '{column}' not recognized as main effect or interaction feature.")


    def get_rbp_position_from_column(self, column=None, binding_fmt=None): 
        assert column.endswith("-shap"), f"Feature name '{column}' must end with '-shap'."
        assert binding_fmt in [True, False,], "binding_fmt must be either True or False."
        
        if binding_fmt:
            suffix = "_binding"
        elif not binding_fmt:
            suffix = ""

        effect_type = self.is_interaction_or_main_effect(column)
        
        if effect_type == "main":
            return column.replace("-main-shap", suffix)
        
        elif effect_type == "interaction":
            feature_names = column.replace("-interaction-shap", "").split("-")
            assert len(feature_names) == 2, f"Interaction feature name '{column}' does not split into two RBP names."
            
            return tuple([feature + suffix for feature in feature_names])
    
    
    def split_rbp_position(self, individual_feature): 
        
        splitter = individual_feature.split("_")
        assert len(splitter) >= 2, f"Feature name '{individual_feature}' does not split into RBP and position."
        assert splitter[1].isdigit() and 1 <= int(splitter[1]) <= 6, f"Second element '{splitter[1]}' is not an integer between 1 and 6 inclusive."    

        # return rbp and position as int
        return splitter[0], int(splitter[1])
    
    
    def get_binding_val_from_metric(self, metric=None):
        assert metric in self.CONFIG["VALID_FEATURE_METRICS"], f"Metric '{metric}' not recognized. Valid metrics are: {self.CONFIG['VALID_FEATURE_METRICS']}"
        assert metric != "Global-SHAP", "Global-SHAP metric does not correspond to a binding value."
        
        if "NOT-Bound" in metric:
            return 0
        elif "Bound" in metric:
            return 1
        else:
            raise ValueError(f"Metric '{metric}' does not correspond to a binding value.")
        

    def retrieve_UBP_ids_for_metric(self, cell_line=None, column=None, metric=None): 
        assert metric in self.CONFIG["VALID_FEATURE_METRICS"], f"Metric '{metric}' not recognized. Valid metrics are: {self.CONFIG['VALID_FEATURE_METRICS']}"
        assert cell_line in self.CONFIG["CELL_LINES"], f"Cell line '{cell_line}' not recognized. Valid cell lines are: {self.CONFIG['CELL_LINES']}"
        assert column.endswith("-shap"), "Column name must end with '-shap'."

        path = f'{self.CONFIG["UBP_ID_DIR"]}/{cell_line}_unique_binding_pattern_ID_reference_table.tsv.gz'

        ubp_id_lf = pl.scan_csv(
            path, 
            separator="\t",
        )

        if "Bound" in metric: 
            effect_type = self.is_interaction_or_main_effect(column)
            binding_cols = self.get_rbp_position_from_column(
                column = column, 
                binding_fmt = True
            )
            
            binding_val = self.get_binding_val_from_metric(metric)
            
            if effect_type == "main":

                subset = ubp_id_lf.select(
                    [self.CONFIG["UBP_COL_NAME"], binding_cols]
                ).filter(
                    # single column
                    pl.col(binding_cols) == binding_val
                )
            elif effect_type == "interaction":
                subset = ubp_id_lf.select(
                    [self.CONFIG["UBP_COL_NAME"]] + list(binding_cols)
                ).filter(
                    (pl.col(binding_cols[0]) == binding_val) & 
                    (pl.col(binding_cols[1]) == binding_val)
                )
        
        elif metric== "Global-SHAP": 
            subset = ubp_id_lf.select(
                [self.CONFIG["UBP_COL_NAME"]]
            )

        else: 
            raise ValueError(f"Metric '{metric}' not recognized.")
        
        unique_ids = subset.collect()[self.CONFIG["UBP_COL_NAME"]].to_list()
        if len(unique_ids) != len(set(unique_ids)):
            raise ValueError(f"Duplicate values found in {self.CONFIG['UBP_COL_NAME']} for cell line '{cell_line}', column '{column}', metric '{metric}'.")
        
        return unique_ids
    

    def update_default_dict(self, default=None, new=None): 
        assert isinstance(default, dict), "default must be a dictionary."
        assert isinstance(new, dict), "new must be a dictionary."
        
        extra_keys = set(new.keys()) - set(default.keys())
        assert not extra_keys, f"Keys {extra_keys} in 'new' are not present in 'default'."
        
        for key in new:
            default[key] = new[key]
        
        return default


    def is_ppi(self, pair, ppi_source=None): 
        assert isinstance(pair, str), "pair must be a string."
        assert ppi_source in self.CONFIG["PPI_SOURCES"], f"ppi_source must be one of {self.CONFIG['PPI_SOURCES']}."
        assert len(pair.split('-')) == 2, "pair must be in following example formats: 'RBP1-RBP2' or 'RBP1_Pos1-RBP2_Pos2'."
        assert " " not in pair, "pair must not contain spaces."

        pair = pair.upper()
        
        if "_" in pair: 
            parts = pair.split('-')
            for part in parts:
                subparts = part.split('_')
                assert subparts[-1].isdigit() and 1 <= int(subparts[-1]) <= 6, f"Element '{subparts[-1]}' is not an integer between 1 and 6 inclusive."

            rbp1 = parts[0].split('_')[0].upper()
            rbp2 = parts[1].split('_')[0].upper()
        
        else: 
            rbp1, rbp2 = pair.split('-')
            rbp1 = rbp1.upper()
            rbp2 = rbp2.upper()

        assert rbp1 != rbp2, "Self-interactions are not allowed."

        interaction = "-".join(
            sorted([rbp1, rbp2])
        )
        assert self.ppi.filter(pl.col("Interaction") == "-".join(sorted([rbp1, rbp2], reverse=True))).height == 0, f"Interaction '{rbp2}-{rbp1}' found in PPI table; interactions should be sorted alphabetically."
        
        filtered = self.ppi.filter(pl.col("Interaction") == interaction)
        assert filtered.height == 1, f"Interaction '{interaction}' not found in PPI table."

        value = filtered[ppi_source].item()
        assert value is None or isinstance(value, bool), f"PPI value for interaction '{interaction}' from source '{ppi_source}' is not None or boolean."
        
        return value
    
    
    def return_ppi_type(self, pair, ppi_source=None):
        # pair is expected to be in the format 'RBP1_Pos1-RBP2_Pos2'
        assert isinstance(pair, str), "pair must be a string."
        assert ppi_source in self.CONFIG["PPI_SOURCES"], f"ppi_source must be one of {self.CONFIG['PPI_SOURCES']}."
        assert len(pair.split('-')) == 2, "pair must be in the format 'RBP1_Pos1-RBP2_Pos2'."
        assert " " not in pair, "pair must not contain spaces."

        part1, part2 = pair.split('-')
        _, pos1 = self.split_rbp_position(part1)
        _, pos2 = self.split_rbp_position(part2)

        ppi_result = self.is_ppi(pair, ppi_source=ppi_source)
        
        if ppi_result is None:
            return None
        elif ppi_result is False:
            return False
        
        elif ppi_result is True:
            if pos1 == pos2:
                return "Same-Position"
            elif pos1 != pos2:
                return "Different-Position"


    def add_ppi_stats_to_long_df(self, df=None): 
        assert isinstance(df, pl.DataFrame), "df must be a polars DataFrame."
        assert df["Column Type"].unique().to_list() == ["interaction"], "Only interaction columns are supported."
        assert not (df["RBP 1"] == df["RBP 2"]).any(), "There are rows where 'RBP 1' equals 'RBP 2'."

        for ppi_source in self.CONFIG["PPI_SOURCES"]:
            df = df.with_columns(
                pl.col("Column").map_elements(
                    lambda pair: str(self.return_ppi_type(pair.replace('-interaction-shap', ''), ppi_source=ppi_source)),
                    return_dtype=pl.String
                ).alias(ppi_source)
            )

            assert df[ppi_source].unique().len() == 4, f"Expected 4 unique values in column '{ppi_source} but got {df[ppi_source].unique().to_list()}."
            assert df[ppi_source].null_count() == 0, f"Null values found in column '{ppi_source}'."

        logger.success("PPI designation columns added to DataFrame.")
        return df


#########################################################
################ ANALYSIS FUNCTIONS #####################
#########################################################


    def calculate_metric_for_column(self, cell_line=None, column=None, metric=None): 
        assert metric in self.CONFIG["VALID_FEATURE_METRICS"], f"Metric '{metric}' not recognized. Valid metrics are: {self.CONFIG['VALID_FEATURE_METRICS']}"
        assert cell_line in self.CONFIG["CELL_LINES"], f"Cell line '{cell_line}' not recognized. Valid cell lines are: {self.CONFIG['CELL_LINES']}"
        assert column.endswith("-shap"), "Feature name must end with '-shap'."

        ubp_ids = self.retrieve_UBP_ids_for_metric(
            cell_line=cell_line,
            column=column,
            metric=metric
        )

        # No UBP ids found for this metric
        if len(ubp_ids) == 0:
            logger.warning(
                f"NOTE: No UBPs for '{metric}' on '{column}' in cell line '{cell_line}'."
            )
            final_value = float("nan")

        else: 

            interaction_values_files = sorted(
                glob.glob(
                    f"{self.CONFIG['INTERACTION_VALUES_DIR']}/{cell_line}_*.feather"
                )
            )

            # Use polars to scan the feather files and select only the required columns
            lf = pl.scan_ipc(interaction_values_files).select(
                [self.CONFIG["UBP_COL_NAME"], column]
            ).filter(
                pl.col(self.CONFIG["UBP_COL_NAME"]).is_in(ubp_ids)
            )

            # Check for duplicate UBP IDs and that all of them were pulled correctly
            all_ubp_ids = lf.select(self.CONFIG["UBP_COL_NAME"]).collect()[self.CONFIG["UBP_COL_NAME"]].to_list()
            if len(all_ubp_ids) != len(set(all_ubp_ids)):
                raise ValueError(f"Duplicate UBP IDs found in the filtered data for cell line '{cell_line}', column '{column}', metric '{metric}'.")
            assert sorted(all_ubp_ids) == sorted(ubp_ids), "Mismatch between retrieved UBP IDs and those in the filtered data."

            if metric.startswith("Signed-"):
                final_value = lf.select(
                        pl.col(column).mean()
                    ).collect().item()
                    
            else: 
                final_value = lf.select(
                    pl.col(column).abs().mean()
                ).collect().item()
            
        return {
                metric: {
                    cell_line: {
                        column: {
                            "Value": final_value,
                            "# UBPs": len(ubp_ids)
                        }
                    }
                }
            }

    
    def calculate_metrics_for_column_range(self, cell_line=None, metric=None, start=None, stop=None, ):
        """
        Calculates the metric for a range of columns for a given cell line using parallel processing.

        Args:
            cell_line (str): The cell line to use.
            metric (str): The metric to calculate.
            start (int): 1-indexed start position in the column list (inclusive).
            stop (int): 1-indexed stop position in the column list (inclusive).

        Returns:
            List of results from calculate_metric_for_column for each column.
        """
        assert cell_line in self.CONFIG["CELL_LINES"], f"Cell line '{cell_line}' not recognized."
        assert metric in self.CONFIG["VALID_FEATURE_METRICS"], f"Metric '{metric}' not recognized."
        assert start >= 1 and stop > start, "Start must be >= 1 and stop must be > start."
        
        num_features = len(self.rbp_feature_metadata[cell_line]["Features"])
        if not ((stop - start + 1) == self.CONFIG["COLS_PER_JOB"] or stop == num_features):
            raise AssertionError(
                f"Column range size must equal COLS_PER_JOB ({self.CONFIG['COLS_PER_JOB']}) unless this is the last chunk ending at the total number of features ({num_features})."
            )

        features = self.rbp_feature_metadata[cell_line]["Features"]
        
        # Convert 1-indexed to 0-indexed python slice
        selected_features = features[start-1:stop]
        assert len(selected_features) == (stop - start + 1), "Selected features length mismatch."
        assert len(selected_features) == len(set(selected_features)), "Duplicate features found in selected range."

        results = []
        with ThreadPoolExecutor(max_workers=self.manually_set_num_tasks()) as executor:
            futures = [
                executor.submit(
                    self.calculate_metric_for_column,
                    cell_line=cell_line,
                    column=feature,
                    metric=metric
                )
                for feature in selected_features
            ]

            desc = f"{cell_line} | {metric} | cols {start}-{stop}"
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc, file=sys.stdout):
                result = future.result()
                results.append(result)
    
        # Ensure all results have the same metric, cell line, and merge feature keys
        merged = {}
        for res in results:
            # Each res is {metric: {cell_line: {column: {...}}}}
            assert len(res) == 1
            metric_key = next(iter(res))
            assert metric_key == metric

            assert len(res[metric_key]) == 1
            cell_line_key = next(iter(res[metric_key]))
            assert cell_line_key == cell_line

            feature_dict = res[metric_key][cell_line_key]
            assert len(feature_dict) == 1
            
            if not merged:
                merged = {metric: {cell_line: {}}}
            merged[metric][cell_line].update(feature_dict)

        num_features = len(merged[metric][cell_line])
        expected_num = stop - start + 1
        assert num_features == expected_num, f"Expected {expected_num} features, got {num_features}"

        output_filename = f"{self.CONFIG['TMP_CACHE_DIR']}/{metric}_{cell_line}_{start}_{stop}.json"
        with open(output_filename, "w") as f:
            json.dump(merged, f, indent=4)
        
        logger.success(f"COMPLETED: Calculated '{metric}' for '{cell_line}' from column {start} to {stop}. Results saved to '{output_filename}'.")


    def retrieve_shap_values_for_metric(self, metric = None): 
        assert metric in self.CONFIG["VALID_FEATURE_METRICS"], f"Metric '{metric}' not recognized."
        OUTPUT_FILE = pathlib.Path(self.CONFIG["SHAP_AVG_DIR"]) / f"{metric}.tsv"

        if OUTPUT_FILE.exists():
            logger.success(f"FROM CACHE: loading '{metric}' values from '{OUTPUT_FILE}'...")
            table = pl.read_csv(OUTPUT_FILE, separator="\t")
            for col in table.columns:
                if col.lower().startswith("position"):
                    table = table.with_columns(pl.col(col).cast(pl.UInt8))
                elif col.startswith("# UBPs - "):
                    table = table.with_columns(pl.col(col).cast(pl.UInt32))
                elif "rec-y2h" in col.lower() or "street et al" in col.lower():
                    table = table.with_columns(pl.col(col).cast(pl.String))
            
            return table

        else:
            logger.info(f"SHAP avg file for '{metric}' not found. Aggregating cached metric files...")
            
            # Check that all expected cache files exist before aggregation
            missing_files = []
            for cell_line in self.CONFIG["CELL_LINES"]:
                features = self.rbp_feature_metadata[cell_line]["Features"]
                chunks = return_parallelization_start_stop(features, self.CONFIG)

                for start, stop in chunks:
                    output_filename = f"{self.CONFIG['TMP_CACHE_DIR']}/{metric}_{cell_line}_{start}_{stop}.json"
                    if not pathlib.Path(output_filename).exists():
                        missing_files.append(output_filename)

            if missing_files:
                raise FileNotFoundError(f"Missing cache files: {missing_files}")

            # Proceed with aggregation after confirming all files exist
            results = []
            cache_files = sorted(glob.glob(f"{self.CONFIG['TMP_CACHE_DIR']}/{metric}_*.json"))

            for cache_file in cache_files:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                # Assert metric is the only key
                assert list(data.keys()) == [metric], f"Expected only metric key in {cache_file}"

                for cell_line, columns in data[metric].items():
                    for column, vals in columns.items():
                        num_ubps = vals["# UBPs"]
                        value = vals["Value"]

                        col_type = self.is_interaction_or_main_effect(column)
                        rbp_info = self.get_rbp_position_from_column(column, binding_fmt=False)

                        if col_type == "main":
                            rbp1, pos1 = self.split_rbp_position(rbp_info)
                            rbp2, pos2 = rbp1, pos1

                        elif col_type == "interaction":
                            (f1, f2) = rbp_info
                            rbp1, pos1 = self.split_rbp_position(f1)
                            rbp2, pos2 = self.split_rbp_position(f2)

                        else: 
                            raise ValueError(f"Column type '{col_type}' not recognized.")

                        results.append({
                            "Cell Line": cell_line,
                            "Column": column,
                            "Column Type": col_type,
                            "Sorted RBP Pair": "-".join(sorted([rbp1, rbp2])),
                            "RBP 1": rbp1,
                            "Position 1": pos1,
                            "RBP 2": rbp2,
                            "Position 2": pos2,
                            f"# UBPs - {metric}": num_ubps,
                            f"Value - {metric}": value
                        })

            df = pl.DataFrame(results)
            df = df.sort(["Cell Line", "Position 1", "Position 2", "RBP 1", "RBP 2"])

            # Check for duplicates in ("Cell Line", "Column")
            dupes = df.group_by(["Cell Line", "Column"]).len(name="count").filter(pl.col("count") > 1)
            if dupes.height > 0:
                raise ValueError(f"Duplicate (Cell Line, Column) pairs found: {dupes}")

            # For each cell line, check number of columns matches number of features
            for cell_line in self.CONFIG["CELL_LINES"]:
                n_features = len(self.rbp_feature_metadata[cell_line]["Features"])
                n_columns = df.filter(pl.col("Cell Line") == cell_line)["Column"].n_unique()
                assert n_columns == n_features, (
                    f"Cell line '{cell_line}' has {n_columns} columns in results but {n_features} features in metadata."
                )   

            df = df.with_columns([
                pl.col("RBP 1").str.to_uppercase(),
                pl.col("RBP 2").str.to_uppercase(), 
                pl.col("Sorted RBP Pair").str.to_uppercase()
            ])

            # Assert that Position 1 is always <= Position 2
            if not (df["Position 1"] <= df["Position 2"]).all():
                raise AssertionError("Found rows where Position 1 > Position 2.")

            # no main effect columns or different position but same RBP columns since those are not distinct-RBP PPIs
            ppi_annotation = df.filter(
                (pl.col("Column Type") == "interaction")
                & (pl.col("RBP 1") != pl.col("RBP 2"))
            )
            ppi_annotation = self.add_ppi_stats_to_long_df(df = ppi_annotation)

            union_col = "Rec-Y2H/Street et al. IP-MS (Union)"
            # Build union column as a list
            union_col_values = []
            for rec_val, ip_val in zip(ppi_annotation["Rec-Y2H"].to_list(), ppi_annotation["Street et al | IP-MS"].to_list()):
                # If either column ends with "-Position", take that string (prefer rec_y2h if both)
                if rec_val.endswith("-Position"):
                    union_col_values.append(rec_val)
                elif ip_val.endswith("-Position"):
                    union_col_values.append(ip_val)
                # If either column is "False", take "False"
                elif rec_val == "False" or ip_val == "False":
                    union_col_values.append("False")
                # All other cases
                else:
                    union_col_values.append("None")

            # Add union column to ppi_annotation
            ppi_annotation = ppi_annotation.with_columns(
                pl.Series(union_col, union_col_values)
            )

            # Keep only "Column" and PPI source columns from ppi_annotation
            ppi_cols = self.CONFIG["PPI_SOURCES"]
            ppi_annotation = ppi_annotation.select(["Column", "Cell Line",] + ppi_cols + [union_col])

            assert ppi_annotation.null_count().sum_horizontal().item() == 0, "Null values found in PPI annotation table."
            assert df.null_count().sum_horizontal().item() == 0, "Null values found in SHAP aggregation table before PPI join."

            # Perform left join and validate row count unchanged
            prev_rows = df.height
            df = df.join(ppi_annotation, on=["Cell Line", "Column"], how="left", validate="1:1", maintain_order="left")
            
            assert df.height == prev_rows, "Row count changed after left join; expected no change."
            # Final validation: assert no nulls in INTER-RBP interaction rows
            assert df.filter(
                (pl.col("Column Type") == "interaction") & 
                (pl.col("RBP 1") != pl.col("RBP 2"))
            ).null_count().sum_horizontal().item() == 0, "Null values found in the aggregated SHAP table."

            df.write_csv(OUTPUT_FILE, separator="\t")
            logger.success(f"SUCCESS: SHAP average file for '{metric}' saved to '{OUTPUT_FILE}'.")


    def convert_long_metric_table_to_symmetric_matrix(self, df=None, metric=None):
        assert isinstance(df, pl.DataFrame), "df must be a polars DataFrame."
        assert metric in self.CONFIG["VALID_FEATURE_METRICS"], f"Metric '{metric}' not recognized."
        assert df["Cell Line"].n_unique() == 1, "Only one cell line allowed in the DataFrame."

        val_col = f"Value - {metric}"
        assert val_col in df.columns, f"Value column '{val_col}' not found in DataFrame."
        num_ubp_col = f"# UBPs - {metric}"
        assert num_ubp_col in df.columns, f"Num UBPs column '{num_ubp_col}' not found in DataFrame."

        # Create string columns for binding features by joining RBP and Position with "_"
        df = df.with_columns([
            (pl.col("RBP 1") + "_" + pl.col("Position 1").cast(pl.Utf8)).alias("Binding Feature 1"),
            (pl.col("RBP 2") + "_" + pl.col("Position 2").cast(pl.Utf8)).alias("Binding Feature 2")
        ])
        
        # Assert no duplicate rows for (Binding Feature 1, Binding Feature 2)
        dupes = df.group_by(["Binding Feature 1", "Binding Feature 2"]).len().filter(pl.col("len") > 1)
        if dupes.height > 0:
            raise ValueError(f"Duplicate rows found for (Binding Feature 1, Binding Feature 2): {dupes}")

        sort_columns = ["Position 1", "Position 2", "RBP 1", "RBP 2"]
        df = df.sort(sort_columns).drop(sort_columns)

        # Only swap rows where "Column Type" is "interaction" to avoid duplicating diagonal elements
        df_swapped = (
            df.filter(pl.col("Column Type") == "interaction")
              .rename({
                  "Binding Feature 1": "Binding Feature 2",
                  "Binding Feature 2": "Binding Feature 1"
              })
              .select(df.columns)
        )

        # Concatenate original and swapped DataFrames to ensure symmetry (without duplicating diagonals)
        df_full = pl.concat([df, df_swapped], how="vertical")

        # Assert that for "interaction" type, each unique "Column" appears exactly twice (symmetric pairs)
        interaction_counts = (
            df_full.filter(pl.col("Column Type") == "interaction")
                   .group_by("Column")
                   .len()
        )
        if not (interaction_counts["len"] == 2).all():
            raise AssertionError("Not all interaction columns have exactly 2 rows in the symmetric matrix.")

        # Assert that for "main" type, each unique "Column" appears exactly once
        main_counts = (
            df_full.filter(pl.col("Column Type") == "main")
                   .group_by("Column")
                   .len()
        )
        if not (main_counts["len"] == 1).all():
            raise AssertionError("Not all main columns have exactly 1 row in the symmetric matrix.")

        pivot_tables = {
            metric: {}
        }
        for pivot_col in [val_col, num_ubp_col]:
            pivot_table = (
                df_full.pivot(
                    values=pivot_col,
                    index="Binding Feature 1",
                    columns="Binding Feature 2"
                )
            )
            
            if pivot_col.startswith("# UBPs"):
                key = "UBPs"
            elif pivot_col.startswith("Value"):
                key = "SHAP"
            pivot_tables[metric][key] = pivot_table

        for table_type, pivot_table in pivot_tables[metric].items():
            # Check for null values in the pivot table
            if pivot_table.null_count().sum_horizontal().item() > 0:
                raise ValueError(f"Null values found in pivot table for {table_type}")

            # If the binding value for the metric is 0, assert no NaNs anywhere
            if metric != "Global-SHAP":
                binding_val = self.get_binding_val_from_metric(metric)
                if binding_val == 0:
                    if pivot_table.drop("Binding Feature 1").select(pl.all().is_nan().sum()).sum_horizontal().item() > 0: 
                        raise AssertionError("NaN values found in pivot table when binding value is 0.")
            else: 
                if pivot_table.drop("Binding Feature 1").select(pl.all().is_nan().sum()).sum_horizontal().item() > 0:
                    raise AssertionError("NaN values found in pivot table for Global-SHAP.")

            # Check that columns and index are the same (excluding the index column itself)
            columns_without_index = [col for col in pivot_table.columns if col != "Binding Feature 1"]
            if not (columns_without_index == pivot_table["Binding Feature 1"].to_list()):
                raise AssertionError("Pivot table columns and index do not match.")

            # Check that index values are increasing by position
            index_series = pivot_table["Binding Feature 1"].to_list()
            max_pos = 1
            for idx in index_series:
                pos = int(idx.split("_")[1])
                if pos < max_pos:
                    raise AssertionError("Index positions are not monotonically increasing.")
                max_pos = max(max_pos, pos)

            # If metric is Global-SHAP, assert all values in num_ubp pivot table are the same
            if metric == "Global-SHAP" and table_type == "UBPs":
                vals = pivot_table.drop("Binding Feature 1").to_numpy().flatten()
                vals = vals[~np.isnan(vals)]
                if not np.all(vals == vals[0]):
                    raise AssertionError("Not all values in num_ubp pivot table are the same for Global-SHAP.")

            # Convert to numpy, drop index column, and check symmetry
            mat = pivot_table.drop("Binding Feature 1").to_numpy()
            if not np.allclose(mat, mat.T, equal_nan=True): 
                raise AssertionError(f"Pivot table for {table_type} is not symmetric.")
            
        for table_type, pivot_table in pivot_tables[metric].items():
            df = pivot_table.to_pandas().set_index("Binding Feature 1")
            pivot_tables[metric][table_type] = df

        return pivot_tables


    def get_RBPs_with_min_1_ppi(self, cell_line=None, ppi_source=None):
        assert cell_line in self.CONFIG["CELL_LINES"], f"Cell line '{cell_line}' not recognized."
        assert ppi_source in self.CONFIG["PPI_SOURCES"], f"PPI mode '{ppi_source}' not recognized."

        filtered = self.ppi.filter(
            (pl.col(ppi_source) == True) &
            (pl.col(f"{cell_line} - Both eCLIP") == True)
        )

        rbps = set()
        for interaction in filtered["Interaction"].to_list():
            parts = interaction.split("-")
            assert len(parts) == 2, f"Interaction '{interaction}' does not split into two RBPs"
            rbps.update(parts)
        
        return rbps
    
    # NOTE: this is a DRAFT function and should not be trusted yet
    def plot_double_triangular_heatmaps(self, matrices = None):
        assert isinstance(matrices, dict), "matrices must be a dictionary."

        # Extract matrices
        shap_matrix = matrices[list(matrices.keys())[0]]["SHAP"]
        ubp_matrix = matrices[list(matrices.keys())[0]]["UBPs"]

        # Ensure index/columns match and are in the same order
        assert (shap_matrix.index == shap_matrix.columns).all()
        assert (ubp_matrix.index == ubp_matrix.columns).all()
        assert (shap_matrix.index == ubp_matrix.index).all()

        # Extract dimensions and Labels
        N = shap_matrix.shape[0]
        feature_labels = shap_matrix.index.tolist()
        
        # Convert to NumPy for efficient plotting
        shap_data = shap_matrix.to_numpy()
        ubps_data = ubp_matrix.to_numpy()

        # --- 2. Create Masks ---
        # SHAP: Keep Lower Triangle + Diagonal (Mask Upper)
        shap_masked = np.ma.masked_where(np.triu(np.ones_like(shap_data, dtype=bool), k=1), shap_data)

        # UBPs: Keep Upper Triangle + Diagonal (Mask Lower)
        ubps_masked = np.ma.masked_where(np.tril(np.ones_like(ubps_data, dtype=bool), k=-1), ubps_data)

        # --- 3. Setup Figure ---
        # We use a large figure size to give the small font a chance to render cleanly
        _, ax = plt.subplots(figsize=(20, 18), dpi=500)
        
        # Calculate offset in data coordinates based on gap percentage
        offset = N * 0.02
        
        # --- 4. Plot Heatmaps ---
        
        # Plot SHAP (Lower Triangle)
        # Extent = (left, right, bottom, top). Note Y is inverted (0 at top).
        img_shap = ax.imshow(
            shap_masked, 
            cmap='bwr', 
            interpolation='nearest',
            origin='upper',
            extent=(0, N, N, 0)
        )

        # Plot UBPs (Upper Triangle)
        # Shifted by 'offset' to the right (+X) and up (-Y) to create the gap
        img_ubps = ax.imshow(
            ubps_masked, 
            cmap='PuBuGn', 
            interpolation='nearest',
            origin='upper',
            extent=(offset, N + offset, N - offset, -offset)
        )

        # --- 5. Configure Axes and Labels ---
        
        # Set plot bounds to include the shifted area
        ax.set_xlim(0, N + offset)
        ax.set_ylim(N, -offset) 
        
        # Generate ticks for EVERY row/column
        # We place ticks in the center of the pixels (0.5, 1.5, ... N-0.5)
        tick_locations = np.arange(N) + 0.5
        
        # Set Ticks
        ax.set_xticks(tick_locations)
        ax.set_yticks(tick_locations)
        
        # Set Labels with HARDCODED fontsize=1
        ax.set_xticklabels(feature_labels, rotation=90, ha='center', fontsize=1)
        ax.set_yticklabels(feature_labels, fontsize=1)
        
        # Remove spines (borders) for a cleaner "floating triangles" look
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Remove tick markers (lines), keep text only
        ax.tick_params(length=0)

        # --- 6. Add Colorbars (Legends) ---
        divider = make_axes_locatable(ax)

        # Left Colorbar (SHAP)
        cax_shap = divider.append_axes("left", size="3%", pad="10%") 
        cbar_shap = plt.colorbar(img_shap, cax=cax_shap)
        cax_shap.yaxis.set_ticks_position('left')
        cax_shap.yaxis.set_label_position('left')
        cbar_shap.set_label('TODO', rotation=90, labelpad=5, fontsize=10)

        # Right Colorbar (UBPs)
        cax_ubps = divider.append_axes("right", size="3%", pad="5%")
        cbar_ubps = plt.colorbar(img_ubps, cax=cax_ubps)
        cbar_ubps.set_label('TODO', rotation=270, labelpad=15, fontsize=10)

        ax.set_title(f"\nNOTE 1: This is draft and should not be trusted as Yogi has not finished validations!\
                     \nNOTE2: Yogi already knows several issues and fixes for improving plot\
                     \n\n ⬇️⬅️ Bottom Left is Bound-Only Signed Mean ⬆️➡️ Top Right is # UBPs for that calculation (REMINDER: symmetric) ", fontsize=20, x=0.45, y=1.02)

        # # --- 7. Draw black grid lines around each cell ---
        # # Draw vertical and horizontal lines to create cell borders
        # for i in range(N + 1):
        #     # Vertical lines
        #     ax.plot([i, i], [0, N], color='black', linewidth=0.1, zorder=10)
        #     # Horizontal lines
        #     ax.plot([0, N], [i, i], color='black', linewidth=0.1, zorder=10)
        # # Also draw grid for the shifted upper triangle
        # for i in range(N + 1):
        #     # Vertical lines for shifted grid
        #     ax.plot([i + offset, i + offset], [-offset, N - offset], color='black', linewidth=0.5, zorder=10)
        #     # Horizontal lines for shifted grid
        #     ax.plot([offset, N + offset], [i - offset, i - offset], color='black', linewidth=0.5, zorder=10)

        plt.tight_layout()
        plt.show()


    def plot_side_by_side_heatmaps(self, matrices=None, ppi_source=None, show=False, **kwargs):

        # --- 1. Set up _DEFAULTS and update with kwargs ---
        _DEFAULTS = {
            "figsize": (18, 8),
            "dpi": 100,
            "cmap_shap": "bwr",
            "highlight": None, 
            "cmap_ubps": "PuBuGn",
            "colorbar_shap_label": "SHAP Value",
            "colorbar_ubps_label": "# UBPs",
            "tick_fontsize": 8,
            "legend_titlesize": 12,
            "title_fontsize": 14,
            "pos_legend_fontsize": 14,
            "legend_loc": "lower center",
            "legend_ncol": 6,
            "legend_bbox_to_anchor": (0.5, -0.02),
            "mask_color": "white",
            'cell_linewidth': 0.1,
            'label_padding': 15, 
            "annotation_width": 1,  # Slightly wider than before
            "annotation_height": 1,
            "annotation_offset": 0.3,  # Single offset value for both row and column annotations
            "row_colors": ['#c51b7d','#e9a3c9','#fde0ef','#e6f5d0','#a1d76a','#4d9221'],
        }
        opts = self.update_default_dict(_DEFAULTS, dict(kwargs) if kwargs else {})
        assert opts["highlight"] in self.CONFIG["PPI_TYPES"] or opts["highlight"] is None, f"highlight type {opts['highlight']} not recognized."

        # --- 2. Extract matrices and assert index/columns match ---
        shap_matrix = matrices[list(matrices.keys())[0]]["SHAP"]
        ubps_matrix = matrices[list(matrices.keys())[0]]["UBPs"]
        metric = list(matrices.keys())[0]

        assert (shap_matrix.index.equals(shap_matrix.columns)), "SHAP matrix index and columns do not match"
        assert (ubps_matrix.index.equals(ubps_matrix.columns)), "UBPs matrix index and columns do not match"
        assert (shap_matrix.index.equals(ubps_matrix.index)), "SHAP and UBPs matrix indices do not match"
        assert (shap_matrix.columns.equals(ubps_matrix.columns)), "SHAP and UBPs matrix columns do not match"
        
        feature_labels = shap_matrix.index.tolist()
        N = len(feature_labels)

        if metric.startswith("Signed-"):
            # For signed metrics, center the colormap at 0
            opts['cmap_center'] = 0
        else:
            opts['cmap_center'] = None

        # --- 2a. Build highlight cells for lower triangle and diagonal ---
        if opts['highlight'] is not None:
            highlight_cells = {}

            for i in range(N):
                for j in range(i + 1):  # includes diagonal
                    if i == j:
                        highlight_cells[(i, j)] = False  # diagonal is always False
                    else:
                        feature_i = feature_labels[i]
                        feature_j = feature_labels[j]
                        pair = f"{feature_i}-{feature_j}"

                        # self-interactions do not count (e.g. RBFOX2-RBFOX2)
                        if self.split_rbp_position(feature_i)[0] == self.split_rbp_position(feature_j)[0]:
                            highlight_cells[(i, j)] = False

                        else: 
                            ppi_type = self.return_ppi_type(pair, ppi_source=ppi_source)

                            if ppi_type is not None:   
                                if type(ppi_type) == bool and ppi_type == False:
                                    highlight_cells[(i, j)] = False

                                elif type(ppi_type) == str:
                                    if opts['highlight'] == "All-Positions": 
                                        highlight_cells[(i, j)] = True
                                    elif opts['highlight'] == ppi_type:
                                        highlight_cells[(i, j)] = True
                                    else:
                                        highlight_cells[(i, j)] = False

                            else:
                                highlight_cells[(i, j)] = False
                                
                    if not highlight_cells[(i, j)]:
                        shap_matrix.iloc[i, j] = np.nan
                        shap_matrix.iloc[j, i] = np.nan  # Symmetric
                        ubps_matrix.iloc[i, j] = np.nan
                        ubps_matrix.iloc[j, i] = np.nan  
    
        # --- 3. Prepare position color annotations for rows and columns ---
        row_colors = opts["row_colors"]
        assert len(row_colors) == 6, "There must be 6 colors for 6 positions"
        row_pos = []
        for label in feature_labels:
            _, pos = self.split_rbp_position(label)
            row_pos.append(pos)
        row_color_map = [row_colors[p-1] for p in row_pos]

        # --- 4. Mask upper triangle (keep diagonal and lower) ---
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)

        # --- 5. Create figure and axes ---
        fig, axes = plt.subplots(1, 2, figsize=opts["figsize"], dpi=opts["dpi"])
        ax1, ax2 = axes

        # --- 6. Plot SHAP heatmap (left) ---
        sns.heatmap(
            shap_matrix,
            ax=ax1,
            mask=mask,
            cmap=opts["cmap_shap"],
            center=opts['cmap_center'],
            square=True,
            linecolor="black", 
            cbar_kws={"shrink": 0.8}  # Shrink colorbar to fit better
        )
        # Set colorbar label above the colorbar
        cbar_shap = ax1.collections[0].colorbar
        cbar_shap.ax.set_ylabel(opts["colorbar_shap_label"], labelpad=30, rotation=0, fontsize=opts["legend_titlesize"])
        cbar_shap.ax.yaxis.set_label_coords(0.5, 1.04)

        ax1.set_title(metric, fontsize=opts["title_fontsize"], y=0.93)
        ax1.set_xticks(np.arange(N) + 0.5)
        ax1.set_yticks(np.arange(N) + 0.5)

        ax1.set_xticklabels(feature_labels, rotation=90, fontsize=opts["tick_fontsize"])
        ax1.set_yticklabels(feature_labels, fontsize=opts["tick_fontsize"])

        ax1.tick_params(axis='both', which='both', length=0, pad=opts["label_padding"])  
        ax1.set_ylabel("")
        ax1.set_xlabel("")
        

        # --- 7. Plot UBPs heatmap (right) ---
        sns.heatmap(
            ubps_matrix,
            ax=ax2,
            mask=mask,
            cmap=opts["cmap_ubps"],
            square=True,
            linecolor="black", 
            cbar_kws={"shrink": 0.8}  # Shrink colorbar to fit better
        )
        # Set colorbar label at the top with labelpad=10
        cbar_ubps = ax2.collections[0].colorbar
        cbar_ubps.ax.set_ylabel(opts["colorbar_ubps_label"], labelpad=20, rotation=0, fontsize=opts["legend_titlesize"])
        cbar_ubps.ax.yaxis.set_label_coords(0.5, 1.04)

        ax2.set_title("UBPs", fontsize=opts["title_fontsize"], y=0.93)
        ax2.set_xticks(np.arange(N) + 0.5)
        ax2.set_yticks(np.arange(N) + 0.5)

        ax2.set_xticklabels(feature_labels, rotation=90, fontsize=opts["tick_fontsize"])
        ax2.set_yticklabels(feature_labels, fontsize=opts["tick_fontsize"])

        ax2.tick_params(axis='both', which='both', length=0, pad=opts["label_padding"])
        ax2.set_ylabel("")
        ax2.set_xlabel("")

        # --- 8. Annotate row and column positions outside the heatmap ---
        # Draw colored rectangles for row positions (left of heatmap)

        annotation_width = opts["annotation_width"]
        annotation_height = opts["annotation_height"]
        for ax in [ax1, ax2]:
            for i, color in enumerate(row_color_map):
                # Row annotation (left of y-labels)
                ax.add_patch(plt.Rectangle(
                    (-annotation_width - opts["annotation_offset"], i), annotation_width, annotation_height, color=color, transform=ax.transData, clip_on=False, linewidth=0
                ))
            # Column annotation (bottom of x-labels)
            for i, color in enumerate(row_color_map):
                ax.add_patch(plt.Rectangle(
                    (i, N + 0.05 + opts["annotation_offset"]), annotation_height, annotation_width, color=color, transform=ax.transData, clip_on=False, linewidth=0
                ))

        # --- 8b. Draw grid rectangles only for diagonal and lower triangle cells ---
        for i in range(N):
            for j in range(i+1):  # Only diagonal and lower triangle
                fill = False
                facecolor = 'none'

                if opts['highlight'] is not None and highlight_cells[(i, j)] != True: 
                    fill = True
                    facecolor = 'lightgray'
                
                rect = plt.Rectangle(
                    (j, i), 1, 1,
                    fill=fill,
                    facecolor=facecolor,
                    edgecolor='black',
                    linewidth=opts['cell_linewidth'],
                    zorder=10
                )
                ax1.add_patch(copy.deepcopy(rect))
                ax2.add_patch(copy.deepcopy(rect))

        # --- 9. Add legend for position colors ---
        legend_handles = []
        for idx, color in enumerate(row_colors):
            patch = mpatches.Patch(color=color, label=f"{idx+1}")
            legend_handles.append(patch)

        fig.legend(
            handles=legend_handles,
            loc=opts["legend_loc"],
            ncol=opts["legend_ncol"],
            bbox_to_anchor=opts["legend_bbox_to_anchor"],
            fontsize=opts["pos_legend_fontsize"],
            frameon=True,
            title="Position", 
            title_fontsize=opts["legend_titlesize"]
        )

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        if show:
            plt.show()
        elif not show: 
            return fig


    def plot_side_by_side_heatmaps_for_min_1_ppi(self, cell_line = None, ppi_source=None, metric=None, **kwargs): 
        assert cell_line in self.CONFIG["CELL_LINES"], f"Cell line '{cell_line}' not recognized."
        assert ppi_source in self.CONFIG["PPI_SOURCES"], f"PPI mode '{ppi_source}' not recognized."
        assert metric in self.CONFIG["VALID_FEATURE_METRICS"], f"Metric '{metric}' not recognized."

        rbps_with_ppi = self.get_RBPs_with_min_1_ppi(cell_line=cell_line, ppi_source=ppi_source)
        logger.info(f"Cell line '{cell_line}' has {len(rbps_with_ppi)} RBPs with at least 1 PPI in mode '{ppi_source}'.")
        
        table = self.retrieve_shap_values_for_metric(metric=metric)
        pivot_tables = self.convert_long_metric_table_to_symmetric_matrix(
            df=table.filter(pl.col("Cell Line") == cell_line),
            metric=metric
        )

        # Filter pivot tables to only include RBPs with at least 1 PPI
        filtered_index = [idx for idx in pivot_tables[metric]["SHAP"].index if idx.split("_")[0] in rbps_with_ppi]
        filtered_columns = [col for col in pivot_tables[metric]["SHAP"].columns if col.split("_")[0] in rbps_with_ppi]

        filtered_matrices = {
            metric: {
                "SHAP": pivot_tables[metric]["SHAP"].loc[filtered_index, filtered_columns],
                "UBPs": pivot_tables[metric]["UBPs"].loc[filtered_index, filtered_columns]
            }
        }

        fig = self.plot_side_by_side_heatmaps(matrices=filtered_matrices, ppi_source=ppi_source, **kwargs)
        highlight_str = kwargs['highlight'] if 'highlight' in kwargs and kwargs['highlight'] is not None else "None"

        fig.suptitle(
            f"{cell_line}\nHighlight: {highlight_str} | RBPs with ≥1 PPI ({ppi_source}) | Metric: {metric}",
            fontsize=26,
            y=1.08
        )
        
        ppi_source_str = ppi_source.translate(
            str.maketrans(" |/()", "_____")
        )
        plt.savefig(
            f"{self.CONFIG['FIGURES']['side_by_side_heatmap_dir']}/{cell_line}_{ppi_source_str}_{metric}_highlight_{highlight_str}_side_by_side_heatmap.png",
            bbox_inches='tight',
            dpi=600
        )
        plt.show()

    
    def plot_roc_and_prc_curves_for_metric(self, metric = None): 
        assert metric in self.CONFIG["VALID_FEATURE_METRICS"], f"Metric '{metric}' not recognized."
        
        table = self.retrieve_shap_values_for_metric(metric=metric)
        # ONLY include INTER-RBP interaction rows as those are only eligible for PPI evaluation
        table = table.filter(
            (pl.col("Column Type") == "interaction")
            & (pl.col("RBP 1") != pl.col("RBP 2"))
        )
        assert table.null_count().sum_horizontal().item() == 0, "Null values found in SHAP table after filtering for interaction rows."

        ppi_sources = [col for col in table.columns if "rec-y2h" in col.lower() or "street et al" in col.lower()]
        ppi_types = self.CONFIG["PPI_TYPES"]
        cell_lines = self.CONFIG["CELL_LINES"]
        curve_types = ["roc", "prc"]
        
        sns.set_palette("Set3")

        summary= []
        for plot_type in ["FEATURE-SPECIFIC", "RBP-SPECIFIC_MAX_VALUE"]:
            fig, axes = plt.subplots(
                nrows=2, ncols=2, figsize=(12, 12), dpi=400,
                sharex=True, sharey=True
            )

            for row_idx, cell_line in enumerate(cell_lines):
                for col_idx, curve_type in enumerate(curve_types):
                    ax = axes[row_idx, col_idx]
                    for ppi_source in ppi_sources: 
                        for ppi_type in ppi_types:
                            # Filter table to only rows where the value of ppi_source column is not "None" string
                            curve_input = table.filter(
                                (pl.col(ppi_source) != "None") & 
                                (pl.col("Cell Line") == cell_line)
                            )
                            assert curve_input[ppi_source].n_unique() ==3, f"Expected 3 unique values in column '{ppi_source}' after filtering, found {curve_input[ppi_source].n_unique()}."

                            # if same or different position only, then you need to subset to only those interaction columns 
                            # or else IT WOULD BE OVERLY HARSH to call QKI_3-RBFOX2_4 as "False PPI" when looking at "Same-Position" only
                            if ppi_type == "Same-Position":
                                curve_input = curve_input.filter(
                                    (pl.col("Position 1") == pl.col("Position 2"))
                                )
                            elif ppi_type == "Different-Position":
                                curve_input = curve_input.filter(
                                    (pl.col("Position 1") != pl.col("Position 2"))
                                )
                            
                            # Same and different positions allowed
                            if ppi_type == "All-Positions":
                                ppi_bool = (curve_input[ppi_source].str.ends_with("-Position"))
                            # Must match specific PPI type
                            else:
                                ppi_bool = (curve_input[ppi_source] == ppi_type)

                            # Add PPI column and Abs. SHAP Value column
                            curve_input = curve_input.with_columns([
                                pl.lit(ppi_bool).alias("PPI"),
                                pl.col(f"Value - {metric}").abs().alias("Abs. SHAP Value")
                            ])
                            
                            # Remove rows where "Abs. SHAP Value" is NaN
                            # This occurs for "Bound-Only" metrics when there was no binding observed
                            curve_input = curve_input.filter(~pl.col("Abs. SHAP Value").is_nan())  
                            # Drop all columns in ppi_sources from curve_input
                            curve_input = curve_input.drop(ppi_sources)

                            if plot_type == "RBP-SPECIFIC_MAX_VALUE":
                                # Create a new column with sorted tuple of RBP 1 and RBP 2
                                curve_input = curve_input.with_columns(
                                    pl.struct(["RBP 1", "RBP 2"]).map_elements(lambda x: "-".join(sorted([x["RBP 1"], x["RBP 2"]]))).alias("RBP_PAIR")
                                )
                                # Sort by Abs. SHAP Value descending and keep only the first occurrence per RBP_PAIR (max value)
                                curve_input = curve_input.sort("Abs. SHAP Value", descending=True).unique(
                                    subset=["RBP_PAIR"], 
                                    keep="first", 
                                    maintain_order=True    
                                )
                                
                            # Sort by "Abs. SHAP Value" descending, then by "PPI" descending
                            curve_input = curve_input.sort(["Abs. SHAP Value", "PPI"], descending=True)
                            
                            # Assert no nulls in the entire dataframe
                            assert curve_input.null_count().sum_horizontal().item() == 0, "Nulls found in the dataframe."

                            # assert at least one row with 0 "Abs. SHAP Value"
                            zero_abs_shap_rows = (curve_input["Abs. SHAP Value"] == 0).sum()
                            assert zero_abs_shap_rows > 0, f"Expected rows with 0 'Abs. SHAP Value', found none for {ppi_source} & {ppi_type}."

                            # Assert that there are no negative values in "Abs. SHAP Value"
                            assert (curve_input["Abs. SHAP Value"] >= 0).all(), "'Abs. SHAP Value' contains negative values."
                            # Assert that there is at least one value above 0 in "Abs. SHAP Value"
                            assert (curve_input["Abs. SHAP Value"] > 0).sum() > 0, "No values above 0 found in 'Abs. SHAP Value' column."

                            # Assert every value in "PPI" is exactly True or False (boolean)
                            assert all(val == True or val == False for val in curve_input["PPI"].to_list()), "Non-boolean values found in 'PPI' column."
                            # Assert that there is at least one True value in the "PPI" column
                            assert curve_input["PPI"].sum() > 0, f"No True values found in 'PPI' column for {ppi_source} & {ppi_type}."

                            ppi_source_str_conversion = ppi_source.translate(
                                str.maketrans(" |/()", "_____")
                            )
                            curve_input.write_csv(
                                f"{self.CONFIG['FIGURES']['prc_roc_curves_dir']}/curve_inputs/{metric}_{plot_type}_{cell_line}_{ppi_source_str_conversion}_{ppi_type}_curve_input_data.tsv",
                                separator="\t"
                            )
                            logger.info(f"Plot Type: {plot_type}, Cell line: {cell_line}, PPI source: {ppi_source}, PPI type: {ppi_type}, Curve: {curve_type}, # points: {curve_input.shape[0]}")

                            y_true = curve_input["PPI"].to_numpy()
                            y_score = curve_input["Abs. SHAP Value"].to_numpy()
                            
                            if curve_type == "roc":
                                fpr, tpr, _ = roc_curve(y_true, y_score)
                                roc_auc = roc_auc_score(y_true, y_score)
                                ax.plot(fpr, tpr, label=f"{ppi_source} & {ppi_type} (AUC={roc_auc:.3f})", alpha=0.7)
                                
                                row_dict = {
                                    "Plot Type": plot_type,
                                    "Cell Line": cell_line,
                                    "PPI Source": ppi_source,
                                    "PPI Type": ppi_type,
                                    "Curve": curve_type,
                                    "AUC": roc_auc, 
                                    "ROC Input Table: # INTER-RBP Interaction Features (Rows)": curve_input.shape[0], 
                                    "# Rows as % of All INTER-RBP Interaction Effects": (curve_input.shape[0] / table.filter(pl.col("Cell Line") == cell_line).shape[0]) * 100,
                                    "ROC Input Table: # True PPIs": curve_input["PPI"].sum(), 
                                    "% Rows w/ True PPI": (curve_input["PPI"].sum() / curve_input.shape[0]) * 100,
                                    "# Rows w/ 0 Abs. SHAP Value": zero_abs_shap_rows,
                                    "% Rows w/ 0 Abs. SHAP Value": (zero_abs_shap_rows / curve_input.shape[0]) * 100
                                }
                                
                                row_dict["Abs. SHAP - Min Val"] = curve_input["Abs. SHAP Value"].min()
                                for thresh in self.CONFIG["SHAP_PERCENTILE_THRESHOLDS"]:
                                    # Use thresh as percentile (0-100) for the top X% highest values
                                    shap_value_at_thresh = np.percentile(curve_input["Abs. SHAP Value"].to_numpy(), thresh)
                                    row_dict[f"Abs. SHAP - {thresh}th Percentile"] = shap_value_at_thresh
                                row_dict["Abs. SHAP - Max Val"] = curve_input["Abs. SHAP Value"].max()

                                for thresh in self.CONFIG["PARTIAL_AUC_THRESHOLDS"]:
                                    partial_auc = roc_auc_score(y_true, y_score, max_fpr=thresh)
                                    row_dict[f"Partial AUC @ FPR={thresh}"] = partial_auc
                                    
                                summary.append(row_dict)

                            else:
                                precision, recall, _ = precision_recall_curve(y_true, y_score)
                                prc_auc = auc(recall, precision)
                                baseline = (curve_input["PPI"].sum() / curve_input.shape[0]) 
                                ax.plot(recall, precision, label=f"{ppi_source} & {ppi_type} (AUC={prc_auc:.3f}) [Baseline: {baseline:.3f}]", alpha=0.5)
                    
                    if curve_type == "roc":
                        ax.plot([0, 1], [0, 1], 'k--', lw=1, label="Baseline")
                        ax.set_xlabel("False Positive Rate", fontsize=12)
                        ax.set_ylabel("True Positive Rate", fontsize=12)
                        ax.set_title(f"ROC Curve - {cell_line}", fontweight='bold', fontsize=16)
                    
                    else:
                        ax.set_xlabel("Recall", fontsize=12)
                        ax.set_ylabel("Precision", fontsize=12)
                        ax.set_title(f"PRC Curve - {cell_line}", fontweight='bold', fontsize=16)
                    
                    if curve_type == "prc":
                        legend_fontsize = 7
                    elif curve_type == "roc":
                        legend_fontsize = 5.5

                    ax.legend(loc="best", fontsize=legend_fontsize, frameon=True)
            
            if plot_type == "RBP-SPECIFIC_MAX_VALUE":
                note = "\nNOTE 7: RBP-Specific takes max SHAP val per unique RBP pair within Same, or Different, or All Positions"
            else: 
                note = ""

            fig.suptitle(
                "\nNOTE 1: PPI status: True (tested & interacts), False (tested & no interaction), and Null (not tested)" + 
                "\nNOTE 2: Null PPI values removed from curve creation to keep only True Positive and True Negatives" +
                "\nNOTE 3: [Only applicable to 'Bound-Only' based metrics] NaN values removed (aka. no binding observed)" +
                '\nNOTE 4: "Same" and "Different" position PPI curves subset to only interactions at same or different positions, respectively' +
                "\nNOTE 5: Curve creation does not include INTRA-RBP interactions (e.g. RBFOX2_3-RBFOX2_4)" +
                "\nNOTE 6: Union PPI is: True (either resource) --> True, then False --> if either resource is False, else None (and hence, removed)" +
                note +
                f"\n\n{plot_type}: ROC and PRC Curves by Cell Line and PPI Source & PPI Type", 
                fontsize=13, y=1.01
            )

            plt.tight_layout()
            plt.savefig(
                f"{self.CONFIG['FIGURES']['prc_roc_curves_dir']}/{plot_type}-{metric}-roc_prc_curves.png",
                bbox_inches='tight',
                dpi=400
            )

            plt.show()

        summary_df = pl.DataFrame(summary).sort("AUC", descending=True)
        summary_df.write_csv(
            f"{self.CONFIG['FIGURES']['prc_roc_curves_dir']}/{metric}_roc_prc_summary_partial_auc_and_value_percentiles.tsv",
            separator="\t"
        )

        return summary_df
    

    def plot_actual_psi_by_binding_for_PPI(self,):
        
        # Collect summary statistics for all plots
        summary_rows = []
        category_names = [
            "Both",
            "XOR_1st_Feature",
            "XOR_2nd_Feature"
        ]
        stat_names = ["n", "mean", "median", "mad"]

        for cell_line in self.CONFIG["CELL_LINES"]:
            first_order_shap_table = pl.scan_ipc(
                f"{self.CONFIG['FIRST_ORDER_SHAP_CACHE_DIR']}/{cell_line}_all-data.feather"
            )
            cols = [col for col in first_order_shap_table.collect_schema().names() if col.endswith("_binding")] + ["Target_PSI"]
            first_order_shap_table = first_order_shap_table.select(cols).collect()

            features = self.rbp_feature_metadata[cell_line]["Features"]
            for column in tqdm(features, desc=f"{cell_line} | PSI by binding features"):
                col_type = self.is_interaction_or_main_effect(column)
                rbp_info = self.get_rbp_position_from_column(column, binding_fmt=True)

                if col_type == "interaction":
                    f1_binding, f2_binding = rbp_info  # e.g., "RBFOX2_3_binding", "QKI_2_binding"
                    rbp1_name, _ = self.split_rbp_position(f1_binding.replace("_binding", ""))
                    rbp2_name, _ = self.split_rbp_position(f2_binding.replace("_binding", ""))

                    # avoiding INTRA-RBP interactions
                    if rbp1_name != rbp2_name:
                        first_rbp, second_rbp = sorted([rbp1_name, rbp2_name])
                        interaction_str = f"{first_rbp}-{second_rbp}"

                        # Filter self.ppi for the exact interaction (should be exactly one row)
                        ppi_row = self.ppi.filter(pl.col("Interaction") == interaction_str)
                        assert ppi_row.height == 1, f"Expected 1 row for interaction '{interaction_str}', found {ppi_row.height}"

                        # For all PPI source columns, get those that are True for this row
                        ppi_sources_true = sorted([col for col in self.CONFIG["PPI_SOURCES"] if ppi_row[col].item() == True])

                        # check that there is PPI in some resource
                        if len(ppi_sources_true) > 0:
                            # Masks and PSI extraction
                            both = first_order_shap_table.filter(
                                (pl.col(f1_binding) == 1) & (pl.col(f2_binding) == 1)
                            ).select("Target_PSI")["Target_PSI"].to_list()
                            first_only = first_order_shap_table.filter(
                                (pl.col(f1_binding) == 1) & (pl.col(f2_binding) == 0)
                            ).select("Target_PSI")["Target_PSI"].to_list()
                            second_only = first_order_shap_table.filter(
                                (pl.col(f1_binding) == 0) & (pl.col(f2_binding) == 1)
                            ).select("Target_PSI")["Target_PSI"].to_list()

                            # need to have examples of both binding
                            if len(both) != 0:
                                plot_data = [both, first_only, second_only]

                                first_feature_name = f1_binding.replace("_binding", "")
                                second_feature_name = f2_binding.replace("_binding", "")
                                labels = [
                                    f"Both\n{first_feature_name} & {second_feature_name}",
                                    f"XOR {first_feature_name}",
                                    f"XOR {second_feature_name}"
                                ]

                                # Stats: (n, mean, median, MAD)
                                stats = []
                                for arr in plot_data:
                                    if len(arr) > 0:
                                        med = float(np.median(arr))
                                        n = int(len(arr))
                                        mean_val = float(np.mean(arr))
                                        mad_val = float(np.median(np.abs(np.asarray(arr) - med)))
                                        stats.append((n, mean_val, med, mad_val))
                                    else:
                                        stats.append((0, float("nan"), float("nan"), float("nan")))

                                # Build a DataFrame for seaborn
                                flat_data = [(lbl, val) for lbl, arr in zip(labels, plot_data) for val in arr]
                                plot_df = pd.DataFrame(flat_data, columns=["Group", "PSI"])

                                # Consistent colors: both, first-only, second-only
                                colors = ['#66c2a5', '#fc8d62', '#8da0cb']

                                plt.figure(figsize=(6, 3), dpi=200)
                                ax = sns.violinplot(
                                    data=plot_df,
                                    x="Group",
                                    y="PSI",
                                    order=labels,
                                    hue="Group",
                                    palette=colors,
                                    cut=0,
                                    legend=False
                                )

                                ax.set_title(
                                    f"PPI found in {len(ppi_sources_true)}: {', '.join(ppi_sources_true)}\n{cell_line} - {first_feature_name} & {second_feature_name}",
                                    fontsize=10, x=0.5
                                )
                                ax.set_xlabel("Binding Category")
                                ax.set_ylabel("Actual PSI")

                                ax.tick_params(axis='x', labelsize=8)

                                # Legend with counts, mean, median, MAD
                                legend_patches = []
                                for lbl, col, (n, mean, median, mad_val) in zip(labels, colors, stats):
                                    new_label = lbl.replace("\n", " ")

                                    legend_patches.append(
                                        mpatches.Patch(
                                            facecolor=col,
                                            label=f"{new_label} n#: {n:,} | Avg: {mean:.3g}",
                                            edgecolor="black",
                                            linewidth=0.7,
                                            alpha=1.0
                                        )
                                    )
                                ax.legend(
                                    handles=legend_patches,
                                    loc="center left",
                                    bbox_to_anchor=(1.01, 0.5),
                                    frameon=True,
                                    fontsize=8,
                                    borderaxespad=0.5,
                                    handleheight=2,
                                    handlelength=2,
                                    labelspacing=1.5,
                                    borderpad=1
                                )

                                plt.tight_layout()
                                plt.savefig(
                                    f"{self.CONFIG['FIGURES']['actual_psi_by_binding_dir']}/all_plots/{cell_line}_{first_feature_name}_{second_feature_name}_PPI_actual_psi_by_binding_violinplot.png",
                                    dpi=200,
                                    bbox_inches='tight'
                                )
                                plt.close()

                                # Collect summary statistics for this plot
                                summary_row = {
                                    "Cell Line": cell_line,
                                    "Interaction Column": column, 
                                    "Feature 1": first_feature_name,
                                    "Feature 2": second_feature_name,
                                }
                                # Use category_names for summary columns
                                for cat_idx, cat in enumerate(category_names):
                                    n, mean, median, mad = stats[cat_idx]
                                    summary_row[f"{cat} - #"] = n
                                    summary_row[f"{cat} - Mean"] = mean
                                    summary_row[f"{cat} - Median"] = median
                                    summary_row[f"{cat} - Median Abs. Dev."] = mad
                                summary_rows.append(summary_row)

        
        # Create polars DataFrame and write as TSV
        summary_df = pl.DataFrame(summary_rows).sort(["Cell Line", "Interaction Column"])
        output_path = f"{self.CONFIG['FIGURES']['actual_psi_by_binding_dir']}/summary_table_for_PPI_ONLY_actual_psi_by_binding.tsv"
        
        summary_df.write_csv(output_path, separator="\t")
        logger.success(f"Summary table written to {output_path}")


    def ppi_vs_non_ppi_shap_value_distributions(self, metric=None):
        assert metric in self.CONFIG["VALID_FEATURE_METRICS"], f"Metric '{metric}' not recognized."

        table = self.retrieve_shap_values_for_metric(metric=metric)
        
        # ONLY include INTER-RBP interaction rows at the same position
        table = table.filter(
            (pl.col("Column Type") == "interaction")
            & (pl.col("RBP 1") != pl.col("RBP 2"))
            & (pl.col("Position 1") == pl.col("Position 2"))
        )
        assert table.null_count().sum_horizontal().item() == 0, "Null values found in SHAP table after filtering for interaction rows."

        for cell_line in self.CONFIG["CELL_LINES"]:
            df_cell = table.filter(
                (pl.col("Cell Line") == cell_line) &
                (~pl.col(f"Value - {metric}").is_nan())
            )

            for ppi_source in self.CONFIG["PPI_SOURCES"]:
                
                # Only keep rows where ppi_source is either "False" or endswith "-Position"
                df_ppi = df_cell.filter(
                    (pl.col(ppi_source) == "False") | (pl.col(ppi_source).str.ends_with("-Position"))
                )
                assert "None" not in df_ppi[ppi_source].unique().to_list(), f"'None' values found in column '{ppi_source}' after filtering."

                for analysis_type in ["RBP-SPECIFIC", "FEATURE-SPECIFIC"]:
                
                    df = df_ppi.with_columns([
                        pl.col(ppi_source).str.ends_with("-Position").alias("PPI"),
                        pl.col(f"Value - {metric}").abs().alias("Abs. SHAP Value"),
                    ])

                    if analysis_type == "RBP-SPECIFIC":
                        df = df.with_columns([
                            pl.struct(["RBP 1", "RBP 2"]).map_elements(lambda x: "-".join(sorted([x["RBP 1"], x["RBP 2"]]))).alias("RBP_PAIR")
                        ])

                        df = (
                            df.sort("Abs. SHAP Value", descending=True)
                            .unique(subset=["RBP_PAIR"], keep="first", maintain_order=True)
                        )

                    df_plot = df.select([
                        "Abs. SHAP Value", "PPI"
                    ]).to_pandas()

                    df_plot["PPI"] = df_plot["PPI"].map({True: "PPI", False: "Non-PPI"})

                    # Now do the plotting after df_plot is created
                    # Example plot (replace with your actual plotting code as needed)
                    plt.figure(figsize=(5,4), dpi=100)
                    
                    ax = sns.violinplot(data=df_plot, x="PPI", y="Abs. SHAP Value", order=["PPI", "Non-PPI"], cut=0)

                    # Calculate max value and set y limit
                    max_val = df_plot["Abs. SHAP Value"].max()
                    y_text = max_val * 1.05

                    # Annotate number of points for each group
                    for i, group in enumerate(["PPI", "Non-PPI"]):
                        n_points = (df_plot["PPI"] == group).sum()
                        ax.text(i, y_text, f"n={n_points}", ha="center", va="bottom", fontsize=8, color="brown")

                    ax.set_ylim(top=y_text * 1.1)  # Add extra space above annotation
                    
                    pairs = [("PPI", "Non-PPI")]
                    annotator = Annotator(ax, pairs, data=df_plot, x="PPI", y="Abs. SHAP Value", order=["PPI", "Non-PPI"])
                    
                    annotator.configure(test='Mann-Whitney-gt', text_format='star', loc='inside', text_offset=1)
                    annotator.apply_test().annotate(line_offset_to_group=1)

                    if analysis_type == "RBP-SPECIFIC":
                        suffix = f"RBP-SPECIFIC means max value across all position combos for a unique RBP pair\n"
                    else:
                        suffix = ""

                    plt.title(
                        "NOTE 1: Only INTER-RBP interactions\n"
                        "NOTE 2: Only SAME POSITION interactions\n"
                        "NOTE 3: Remove NaN values (no binding observed)\n"
                        "NOTE 4: not tested PPIs are removed\n"
                        "NOTE 5: Mann Whitney U test used for PPI > Non-PPI\n"
                        f"{suffix}\n"
                        f"{cell_line}: {ppi_source} ({analysis_type})", fontsize=8)
                    plt.ylabel(f"|{metric}|", fontsize=6,)
                    plt.xlabel("PPI Status", fontsize=10, x=0.4)

                    plt.tight_layout()
                    plt.show()


    def ppi_type_shap_value_distributions(self, metric=None):
        assert metric in self.CONFIG["VALID_FEATURE_METRICS"], f"Metric '{metric}' not recognized."

        table = self.retrieve_shap_values_for_metric(metric=metric)
        
        for ppi_source in self.CONFIG["PPI_SOURCES"]:
            df = table.filter(
                (~pl.col(f"Value - {metric}").is_nan()) &
                ((pl.col(ppi_source) == "False") | (pl.col(ppi_source).str.ends_with("-Position")))
            )

            # Only keep relevant PPI types (already filtered, just rename)
            df = df.rename({ppi_source: "PPI Type"})

            df_plot = df.select([
                "Cell Line",
                f"Value - {metric}",
                "PPI Type"
            ]).to_pandas()

            # Use absolute value for the value column
            df_plot[f"Abs. Value - {metric}"] = df_plot[f"Value - {metric}"].abs()
            # Replace "-" with " " and "False" with "Not PPI" in PPI Type
            df_plot["PPI Type"] = df_plot["PPI Type"].replace("False", "Not PPI").str.replace("-", " ", regex=False)

            x_order = ["HepG2", "K562"]
            hue_order = ["Same Position", "Different Position", "Not PPI"]

            plt.figure(figsize=(7,4), dpi=300)
            ax = sns.violinplot(
                data=df_plot,
                x="Cell Line",
                y=f"Abs. Value - {metric}",
                order = x_order,
                hue="PPI Type",
                hue_order=hue_order,
                cut=0,
                split=False, 
            )

            # Annotate number of points above each distribution
            max_val = df_plot[f"Abs. Value - {metric}"].max()
            y_text = max_val * 1.02
            # Get the positions for each (x, hue) combination
            for i, cell_line in enumerate(x_order):
                for j, ppi_type in enumerate(hue_order):
                    # Find the number of points for this group
                    n_points = len(df_plot[(df_plot["Cell Line"] == cell_line) & (df_plot["PPI Type"] == ppi_type)])
                    # Calculate the x position: distribute hues evenly within each x tick
                    x_pos = i + (j-1) / (len(hue_order) + 1)
                    # For grouped violinplot, the positions are: 0,1,2 for HepG2, 3,4,5 for K562
                    ax.text(x_pos, y_text, f"n={n_points}", ha="center", va="bottom", fontsize=8, color="brown")

            ax.set_ylim(top=y_text * 1.1)  # Add extra space above annotation

            # Move legend to outside center right
            ax.legend(
                title="PPI Type",
                bbox_to_anchor=(1.02, 0.5),
                loc="center left",
                borderaxespad=0.,
                frameon=True
            )

            ax.set_title(f"{ppi_source}: Distributions of different PPI Types across Cell Lines", fontsize=8)
            ax.set_ylabel(f"|{self.CONFIG['LATEX_SYMBOLS'][metric]}|", fontsize=14)
            ax.set_xlabel("Cell Line")

            plt.suptitle(
                "NOTE 1: Only INTER-RBP interactions\n"
                "NOTE 2: NaN values removed (no binding observed)\n"
                "NOTE 3: not tested PPIs are removed\n"
                "NOTE 4: Mann Whitney U tests 'Same Position' > other category\n",
                fontsize=8, y=.95, x=0.4
            )

            pairs = [
                (("HepG2", "Same Position"), ("HepG2", "Different Position")),
                (("HepG2", "Same Position"), ("HepG2", "Not PPI")),
                (("K562", "Same Position"), ("K562", "Different Position")),
                (("K562", "Same Position"), ("K562", "Not PPI")),
            ]
            annotator = Annotator(
                ax,
                pairs,
                data=df_plot,
                x="Cell Line",
                y=f"Abs. Value - {metric}",
                hue="PPI Type",
                order = x_order,
                hue_order=hue_order, 
            )
        
            annotator.configure(test='Mann-Whitney-gt', text_format='star', loc='inside', text_offset=0.8, color='red')
            annotator.apply_test().annotate(line_offset_to_group=0.99)

            # Remove upper axis spine for a cleaner look
            ax.spines['top'].set_visible(False)

            plt.tight_layout()
            plt.show()


    def retrieve_importance_network(self,): 
        OUTPUT_FILE = self.CONFIG["IMPORTANCE_NETWORKS"]

        if pathlib.Path(OUTPUT_FILE).exists():
            logger.success(f"FROM CACHE: loading importance network(s) from '{OUTPUT_FILE}'...")
            
            with open(OUTPUT_FILE, "rb") as f:
                graphs_by_cell_line = pickle.load(f)
            
            return graphs_by_cell_line
        
        else: 
            logger.info("Creating importance network and caching NetworkX object...")

            metric = "Bound-Only"
            val_col = f"Value - {metric}"

            table = self.retrieve_shap_values_for_metric(metric=metric)
            table = (
                table.with_columns(
                    [
                        (pl.col("RBP 1") + "_" + pl.col("Position 1").cast(pl.Utf8)).alias("Feature 1"),
                        (pl.col("RBP 2") + "_" + pl.col("Position 2").cast(pl.Utf8)).alias("Feature 2"),
                    ]
                )
            )

            # Keep only interaction-type edges with weight > 0 (drop NaN and 0 values)
            edges = (
                table.filter(
                    (pl.col("Column Type") == "interaction") &
                    (~pl.col(val_col).is_nan()) & # Exclude NaN values 
                    (pl.col(val_col) > 0) # Exclude 0 values
                ).select(
                    ["Cell Line", "Feature 1", "Feature 2", val_col]
                )
            )

            # Ensure there are no duplicate undirected edges per cell line
            edges = edges.with_columns(
                pl.when(pl.col("Feature 1") <= pl.col("Feature 2"))
                .then(pl.col("Feature 1"))
                .otherwise(pl.col("Feature 2"))
                .alias("_u"),
                pl.when(pl.col("Feature 1") <= pl.col("Feature 2"))
                .then(pl.col("Feature 2"))
                .otherwise(pl.col("Feature 1"))
                .alias("_v"),
            )
            dupes = edges.group_by(["Cell Line", "_u", "_v"]).len().filter(pl.col("len") > 1)
            if dupes.height > 0:
                raise ValueError(f"Duplicate undirected edges detected:\n{dupes}")

            edges_pd = edges.select(["Cell Line", "Feature 1", "Feature 2", val_col]).to_pandas()

            graphs_by_cell_line = {}
            for cell_line in edges_pd["Cell Line"].unique().tolist():
                # Subset to this cell line and deep-copy to avoid any view/copy pitfalls
                df_cl = edges_pd[edges_pd["Cell Line"] == cell_line][["Feature 1", "Feature 2", val_col]].copy(deep=True)

                assert df_cl[val_col].isnull().sum() == 0, f"NaN values found in '{val_col}' for cell line '{cell_line}'."
                assert df_cl[val_col].dtype == float, f"Non-numeric values found in '{val_col}' for cell line '{cell_line}'."

                # Rename the SHAP value column to a generic edge attribute name before creating the graph
                df_cl = df_cl.rename(columns={val_col: "weight"})

                # Much faster than iterating row-by-row
                G = nx.from_pandas_edgelist(
                    df_cl,
                    source="Feature 1",
                    target="Feature 2",
                    edge_attr='weight',
                    create_using=nx.Graph(),
                )

                # Add per-node attributes derived from the node name (e.g., "RBFOX2_3" --> RBP: "RBFOX2", Position: 3)
                rbp_attr = {}
                pos_attr = {}
                avg_edge_weight_attr = {}
                for node in G.nodes:
                    rbp, pos = self.split_rbp_position(node)
                    rbp_attr[node] = rbp
                    pos_attr[node] = pos

                    # Calculate average edge weight for this node
                    edge_weights = [G[node][neighbor]['weight'] for neighbor in G.neighbors(node)]
                    avg_edge_weight_attr[node] = float(np.mean(edge_weights)) if edge_weights else float('nan')

                nx.set_node_attributes(G, rbp_attr, "RBP")
                nx.set_node_attributes(G, pos_attr, "Position")
                nx.set_node_attributes(G, avg_edge_weight_attr, "Average Edge Weight")

                graphs_by_cell_line[cell_line] = {}
                graphs_by_cell_line[cell_line]['graph'] = G
                
                logger.info(f"Cell Line: {cell_line} | # Nodes: {G.number_of_nodes():,} | # Edges: {G.number_of_edges():,}")
                
                graphs_by_cell_line[cell_line]['weight_matrix'] = nx.to_pandas_adjacency(G, weight='weight', nonedge=float("nan"))
        
            with open(OUTPUT_FILE, "wb") as f:
                pickle.dump(graphs_by_cell_line, f)

            logger.success(f"SUCCESS: Importance network(s) cached to '{OUTPUT_FILE}'.")


    def plot_edge_weights_per_position(self): 
        metric = "Bound-Only"
        val_col = f"Value - {metric}"

        # 1) Load the importance network
        graphs_by_cell_line = self.retrieve_importance_network()

        # 2) Build a long-form table: one row per (cell_line, position, edge_weight)
        rows = []
        for cell_line in self.CONFIG["CELL_LINES"]:
            G = graphs_by_cell_line[cell_line]['graph']

            for pos in self.CONFIG["POSITIONS"]:
                assert type(pos) == int and (1<=pos<=6), f"Position '{pos}' is not an integer between 1 and 6 (inclusive)."
                
                # Iterate through edges and collect weights for this position
                for u, v, edge_data in G.edges(data=True):
                    # Extract positions from node names
                    _, pos_u = self.split_rbp_position(u)
                    _, pos_v = self.split_rbp_position(v)
                    
                    # If either node contains this position, include the edge weight
                    if pos_u == pos or pos_v == pos:
                        rows.append(
                            {
                                "Cell Line": cell_line,
                                "Position": pos,
                                val_col: float(edge_data['weight']),
                            }
                        )
                    
        out_df = pl.DataFrame(rows).with_columns(
            [
                pl.col("Cell Line").cast(pl.String),
                pl.col("Position").cast(pl.UInt8),
                pl.col(val_col).cast(pl.Float64),
            ]
        )
        assert (out_df[val_col].null_count() == 0) and \
            (out_df[val_col] != 0).all() and \
            (~out_df[val_col].is_nan()).all(), f"'{val_col}' contains null, NaN, or zero values."


        # --- Plotting ---
        df_plot = out_df.to_pandas()

        x_order = self.CONFIG["CELL_LINES"]
        hue_order = self.CONFIG["POSITIONS"]
        latex_symbol = self.CONFIG["LATEX_SYMBOLS"][metric]
        colors = ["lightgreen", "plum", "darkorange", "yellow", "lightblue", "sandybrown"]

        plot_notes = (
            "NOTE 1: Main effects excluded; INTER-RBP && INTRA-RBP interactions included\n"
            "NOTE 2: Excludes NaN values (no binding observed)\n"
            "NOTE 3: 0 edge weights are considered 'No Edge'\n"
            "NOTE 4: Each edge can contribute to multiple positions if the 2 nodes have different positions\n"
            "\n"
        )

        # 1) Violinplot: Cell Line (x) with hue=Position
        plt.figure(figsize=(8, 4.7), dpi=300)
        ax = sns.violinplot(
            data=df_plot,
            x="Cell Line",
            y=val_col,
            order=x_order,
            hue="Position",
            hue_order=hue_order,
            cut=0,
            linewidth = 0.3, 
            palette=colors
        )

        # Annotate number of points above each (Cell Line, Position) violin
        max_val = float(df_plot[val_col].max())
        y_text = (max_val * 1.02)

        n_hues = len(hue_order)
        group_width = 0.8  # seaborn categorical width
        step = group_width / n_hues

        for i, cl in enumerate(x_order):
            for j, pos in enumerate(hue_order):

                subsetting = df_plot[(df_plot["Cell Line"] == cl) & (df_plot["Position"] == pos)]
                n_points = subsetting.shape[0]
                percent_zero = (subsetting[subsetting[val_col] == 0].shape[0] / n_points) * 100

                # Approximate center of each dodged violin within the category
                x_pos = i - (group_width / 2) + (step / 2) + j * step

                ax.text(
                    x_pos,
                    y_text,
                    f"{n_points:,}", #\n{percent_zero:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="black",
                    clip_on=False,
                )
        
        ax.text(-.43, y_text-0.01, "n: ", ha="center", va="bottom", fontsize=8, color="blue")# \n% 0: ", ha="center", va="bottom", fontsize=8, color="blue")

        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=14)

        ax.set_title(f"{plot_notes}{latex_symbol} ({metric}): Edge Weight distributions", fontsize=10, y=1.07)
        ax.set_xlabel("Cell Line", fontweight='bold', fontsize=16)
        ax.set_ylabel(latex_symbol, fontweight='bold', fontsize=20)

        # Turn off top and bottom spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(
            title="Position",
            bbox_to_anchor=(.45, 0.6),
            loc="center left",
            borderaxespad=0.0,
            frameon=True,
        )
        plt.tight_layout()
        plt.show()

        # 2) Barplot: mean per (Cell Line, Position)
        mean_df = (
            out_df.group_by(["Cell Line", "Position"])
            .agg(pl.col(val_col).mean().alias(f"Mean {val_col}"))
            .sort(["Cell Line", "Position"])
            .to_pandas()
        )

        plt.figure(figsize=(8.5, 5), dpi=300)
        ax = sns.barplot(
            data=mean_df,
            x="Cell Line",
            y=f"Mean {val_col}",
            order=x_order,
            hue="Position",
            hue_order=hue_order,
            errorbar=None,
            palette=colors,
            edgecolor="black",
            linewidth=1
        )

        # Add numeric value labels above each bar (same x, y + 0.0005)
        y_offset = 0.001
        for patch in ax.patches:
            height = patch.get_height()
            if height == 0:
                continue  # Skip bars with height 0

            x = patch.get_x() + patch.get_width() / 2.0
            ax.text(
                x,
                height + y_offset,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                clip_on=False,
            )

        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=14)

        ax.set_title(f"{plot_notes}{latex_symbol} ({metric}): Mean(Edge Weight)", fontsize=10, y=1.09)
        ax.set_xlabel("Cell Line", fontweight='bold', fontsize=16)
        ax.set_ylabel(f"Mean({latex_symbol})", fontweight='bold', fontsize=20)

        # Turn off top and bottom spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(
            title="Position",
            bbox_to_anchor=(.48, 0.7),
            loc="center left",
            borderaxespad=0.0,
            frameon=True,
        )
        plt.tight_layout()
        plt.show()


    def plot_average_importance_between_positions(self): 
        metric = "Bound-Only"
        val_col = f"Value - {metric}"

        # 1) Load the importance network
        graphs_by_cell_line = self.retrieve_importance_network()

        for cell_line in self.CONFIG["CELL_LINES"]:
            G = graphs_by_cell_line[cell_line]['graph']

            # Build matrices: iterate through each pair of positions
            avg_matrix = pd.DataFrame(
                np.nan,
                index=self.CONFIG["POSITIONS"],
                columns=self.CONFIG["POSITIONS"],
                dtype=float
            )
            count_matrix = pd.DataFrame(
                0,
                index=self.CONFIG["POSITIONS"],
                columns=self.CONFIG["POSITIONS"],
                dtype=int
            )

            # For each position pair, collect edge weights
            for pos1 in self.CONFIG["POSITIONS"]:
                for pos2 in self.CONFIG["POSITIONS"]:
                    weights = []
                    
                    # Check each edge in the graph
                    for u, v, edge_data in G.edges(data=True):
                        _, pos_u = self.split_rbp_position(u)
                        _, pos_v = self.split_rbp_position(v)
                        
                        # Include edge if it connects pos1 and pos2
                        if (pos_u == pos1 and pos_v == pos2) or (pos_u == pos2 and pos_v == pos1):
                            weights.append(float(edge_data['weight']))
                    
                    avg_matrix.loc[pos1, pos2] = float(np.mean(weights))
                    count_matrix.loc[pos1, pos2] = len(weights)

            # Assert no nulls in avg_matrix and no 0 values in count_matrix
            assert not pd.isnull(avg_matrix.values).any(), "Null values found in avg_matrix"
            assert not (count_matrix.values == 0).any(), "Zero values found in count_matrix"
            # Assert that both matrices are symmetric
            assert np.allclose(avg_matrix.values, avg_matrix.values.T, equal_nan=True), "avg_matrix is not symmetric"
            assert np.allclose(count_matrix.values, count_matrix.values.T, equal_nan=True), "count_matrix is not symmetric"

            # --- Plotting: 2 subplots side by side ---
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)
            importance_symbol = self.CONFIG["LATEX_SYMBOLS"][metric]
            
            # Mask for lower triangle + diagonal
            mask = np.triu(np.ones_like(avg_matrix, dtype=bool), k=1)
            
            # --- Plot 1: avg_matrix (lower triangle + diagonal) ---
            sns.heatmap(
                avg_matrix,
                ax=ax1,
                mask=mask,
                cmap="Blues",
                cbar_kws={"label": f"Mean({importance_symbol})", "shrink": 0.9},
                square=True, 
                annot=True,
                annot_kws={"fontsize":6},
                fmt = ".4f", 
            )
            cbar1 = ax1.collections[0].colorbar
            cbar1.ax.set_ylabel(f"Mean({importance_symbol})", rotation=0, labelpad=20, fontsize=13)
            cbar1.ax.yaxis.set_label_coords(0.5, 1.07)
            ax1.set_title(f"Mean(Importance Scores)", fontsize=14, y=1.01)
        
            # --- Plot 2: count_matrix (lower triangle + diagonal) ---
            sns.heatmap(
                count_matrix,
                ax=ax2,
                mask=mask,
                cmap="Oranges",
                cbar_kws={"label": "Count", "shrink": 0.9},
                square=True, 
                annot=True,
                annot_kws={"fontsize":6},
                fmt = ","
            )
            cbar2 = ax2.collections[0].colorbar
            cbar2.ax.set_ylabel("#", rotation=0, labelpad=20, fontsize=13)
            cbar2.ax.yaxis.set_label_coords(0.5, 1.07)
            ax2.set_title("# Non-Null Interactions\n(Divisor for Average)", fontsize=14, y=.97,)
            
            n = avg_matrix.shape[0]
            # Add transparent black boxes around unmasked cells
            for row in range(n):
                for col in range(n):
                    if not mask[row, col]:
                        rect = plt.Rectangle((col, row), 1, 1, fill=False, edgecolor='black', linewidth=1)
                        rect2 = copy.deepcopy(rect)
                        
                        ax1.add_patch(rect)
                        ax2.add_patch(rect2)

            for current_ax in [ax1, ax2]:
                current_ax.set_xlabel("")
                current_ax.set_ylabel("")
                current_ax.tick_params(which='both', labelsize=14) 

            plot_notes = (
                "NOTE 1: Main effects excluded; INTER-RBP && INTRA-RBP interactions included\n"
                "NOTE 2: Excludes NaN values (no binding observed)\n"
                "NOTE 3: 0 edge weights are considered 'No Edge'\n"
                "\n"
            )

            fig.suptitle(
                f"{plot_notes}{cell_line}: Average of {importance_symbol} (Importance Scores) Between Positions\n",
                fontsize=10,
                y=1, 
                color='darkgreen'
            )
            fig.supxlabel("Position", fontsize=16, fontweight='bold', x=0.5, y=-0.03, ha='center')
            fig.supylabel("Position", fontsize=16, fontweight='bold', x=0.03)

            plt.tight_layout()
            plt.show()


    def top_interactions_per_cell_line(self): 
        val_col = "Value - Signed-Local-SHAP-Mean-Bound-Only"
        latex_symbol = self.CONFIG["LATEX_SYMBOLS"][f"{val_col.split(' - ')[-1]}"]

        # Retrieve SHAP values for the specified metric
        raw_table = self.retrieve_shap_values_for_metric(metric="Signed-Local-SHAP-Mean-Bound-Only").with_columns(
            pl.col("Column").str.replace("-interaction-shap", "").alias("Cleaned Interaction Name")
        )

        # Filter for interaction columns and sort by absolute value
        table = raw_table.filter(
            (pl.col("Column Type") == "interaction") & 
            (~pl.col(val_col).is_nan()) & 
            (pl.col(val_col) != 0)
        ).with_columns(
            pl.col(val_col).abs().alias("abs_value"),
        ).sort("abs_value", descending=True)

        # Take the top 20 interactions per cell line
        top_interactions = table.group_by("Cell Line").head(20)
        assert top_interactions.height == 40, f"Expected 40 rows (20 per cell line), got {top_interactions.height} rows."

        if top_interactions.select(pl.col("Column")).n_unique() == top_interactions.height:
            logger.warning("\n\nAll top interactions are unique across cell lines.\n\n")

        # Get the top 20 interactions per cell line and get the other cell lines values as well
        both_cell_lines = raw_table.filter(
            pl.col("Column").is_in(
                top_interactions["Column"].unique().to_list()
            )
        )

        # Plotting
        unique_cell_lines = self.CONFIG["CELL_LINES"]
        plot_notes = (
            "NOTE 1: Main effects excluded; INTER-RBP && INTRA-RBP interactions included\n"
            "NOTE 2: Excludes NaN values (no binding observed)\n"
            "NOTE 3: 0 edge weights are considered 'No Edge'\n"
            "\n"
        )

        for mode in ["Per Cell Line", "Both Cell Lines"]:
            
            if mode == "Per Cell Line":
                fig, axes = plt.subplots(1, 2, figsize=(11, 7.5), dpi=300, sharex=True)
                plot_data_list = [
                    (ax, cell_line, top_interactions.filter(pl.col("Cell Line") == cell_line).to_pandas())
                    for ax, cell_line in zip(axes, unique_cell_lines)
                ]
            else:
                both_df = both_cell_lines.to_pandas().copy(deep=True)
                both_df = both_df.sort_values(by=val_col , key=lambda x: x.abs(), ascending=False)
                both_df = both_df.reset_index(drop=True)

                fig, ax = plt.subplots(figsize=(8, 20), dpi=300)
                plot_data_list = [(ax, "", both_df)]

            
            for ax, cell_line, plot_df in plot_data_list:
                
                if mode == "Per Cell Line":
                    colors = ['tomato' if value > 0 else 'dodgerblue' for value in plot_df[val_col]]
                
                sns.barplot(
                    data=plot_df,
                    x=val_col,
                    y="Cleaned Interaction Name",
                    hue = "Cell Line" if mode == "Both Cell Lines" else None,
                    ax=ax,
                    palette=colors if mode == "Per Cell Line" else None,
                    orient="h",
                    edgecolor='black'
                )

                ax.axvline(0, color='black', linestyle='-', linewidth=2)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if mode == "Per Cell Line":
                    ax.spines['left'].set_visible(False)

                ax.tick_params(axis='y', which='both', left=False)
                ax.set_title(f"{cell_line}", pad=20, fontsize=18)
                ax.set_xlabel("")
                ax.set_ylabel("")

                ax.grid(axis='x', linestyle='--', alpha=0.7)
                if mode == "Both Cell Lines":
                    ax.grid(axis='y', linestyle='--', alpha=0.3)

                # Change y-axis tick label colors based on PPI status
                for i, interaction in enumerate(plot_df["Column"].to_list()):
                    rbps = []
                    for feature in self.get_rbp_position_from_column(interaction, binding_fmt=False):
                        rbp, _ = self.split_rbp_position(feature)
                        rbps.append(rbp)
                    if len(set(rbps)) == 2:
                        is_ppi_result = self.is_ppi(
                            interaction.replace("-interaction-shap", ""),
                            ppi_source="Street et al | IP/SEC-MS (Both)"
                        )
                        if is_ppi_result:
                            ax.get_yticklabels()[i].set_color('magenta')

            fig.suptitle(
                f"{plot_notes}{latex_symbol}: Top 20 Interactions per Cell Line",
                fontsize=16,
                y=0.98,
                x=0.53
            )
            fig.supxlabel(latex_symbol, fontsize=24, y=0.04, x=0.52)

            plt.tight_layout()
            plt.show()
    

    def plot_interaction_cell_line_similarity_scatterplot(self, label_top_points = None): 
        assert type(label_top_points) == int, f"'label_top_points' must be an integer. Got '{type(label_top_points)}'."

        val_col = f"Value - Signed-Local-SHAP-Mean-Bound-Only"
        latex_symbol = self.CONFIG["LATEX_SYMBOLS"]["Signed-Local-SHAP-Mean-Bound-Only"]
        ppi_col = "Street et al | IP/SEC-MS (Both)"

        plot_types = {
            "Main & Interaction": {
                'hue': "Column Type",
                'palette': {"Main": 'darkorange', "Interaction": 'skyblue'},
                'hue_order': ["Main", "Interaction"],
                'supplementary_note': (
                    "NOTE: subsets to MAIN & INTERACTION effects for RBPs w/ eCLIP in both cell lines\n"
                )
            },
            "Interaction Only": {
                'hue': "PPI Type",
                'palette': {"PPI": 'magenta', "Not PPI": 'green'},
                'hue_order': ["PPI", "Not PPI"],
                'supplementary_note': (
                    "NOTE: subsets to INTERACTION effects for RBPs w/ eCLIP in both cell lines\n"
                )
            },
        }

        plot_notes = (
            f"NOTE: Top {label_top_points} highest & lowest 'cell line average' points are labeled\n"
            f"NOTE: Points with NaN values in either cell line (aka. no binding observed) are not plotted\n"
        )

        for plot_type in plot_types.keys():
            # Load the table
            table = self.retrieve_shap_values_for_metric(metric=val_col.split(" - ")[-1])
            if plot_type == "Interaction Only":
                table = table.filter(pl.col("Column Type") == "interaction")

            # Filter for rows that appear in both cell lines
            column_counts = table.group_by("Column").agg(pl.col("Cell Line").n_unique().alias("n_cell_lines"))
            assert column_counts.filter(pl.col("n_cell_lines") > 2).height == 0, \
                "Some columns appear in more than 2 cell lines"
            
            columns_in_both = column_counts.filter(pl.col("n_cell_lines") == 2)["Column"].to_list()

            table = table.filter(pl.col("Column").is_in(columns_in_both))
            assert table.group_by("Column").agg(pl.col("Cell Line").n_unique()).filter(pl.col("Cell Line") != 2).height == 0, \
                "Some columns do not have exactly one row per cell line"
            
            # Pivot to wide format: columns as rows, cell lines as columns
            pivot_table = table.select(["Column", "Cell Line", val_col]).pivot(
                values=val_col,
                index="Column",
                columns="Cell Line", 
                sort_columns=True,
            ).sort("Column")

            pivot_df = pivot_table.to_pandas().reset_index()

            # Get Column Type information from original table for each column
            col_type_info = table.select(["Column", "Column Type"]).unique().to_pandas()
            col_type_dict = dict(zip(col_type_info["Column"], col_type_info["Column Type"]))
            pivot_df["Column Type"] = pivot_df["Column"].map(col_type_dict)
            pivot_df["Column Type"] = pivot_df["Column Type"].str.capitalize()

            # Add PPI Info for each column 
            ppi_type_info = table.select(["Column", ppi_col]).unique().to_pandas()
            ppi_type_dict = dict(zip(ppi_type_info["Column"], ppi_type_info[ppi_col]))
            pivot_df["PPI Type"] = pivot_df["Column"].map(ppi_type_dict)
            pivot_df["PPI Type"] = pivot_df["PPI Type"].apply(lambda x: "PPI" if isinstance(x, str) and x.endswith("-Position") else "Not PPI")

            # Calculate average SHAP value across cell lines for ranking
            pivot_df["Average_Value"] = (pivot_df["HepG2"].fillna(0) + pivot_df["K562"].fillna(0)) / 2
            
            # Identify top 20 highest and top 20 lowest
            top_n_high = pivot_df.nlargest(label_top_points, "Average_Value").index.tolist()
            top_n_low = pivot_df.nsmallest(label_top_points, "Average_Value").index.tolist()
            annotate_indices = set(top_n_high + top_n_low)

            # Calculate statistics
            n_total_points = len(pivot_df)
            
            # Calculate correlations (excluding NaN values)
            valid_mask = ~(pivot_df["HepG2"].isna() | pivot_df["K562"].isna())
            valid_data = pivot_df[valid_mask]
            n_valid_points = len(valid_data)

            pearson_r, _ = pearsonr(valid_data["HepG2"], valid_data["K562"])
            spearman_r, _ = spearmanr(valid_data["HepG2"], valid_data["K562"])
        
            # Calculate zero and NaN statistics
            both_zero_valid = ((valid_data["HepG2"] == 0) & (valid_data["K562"] == 0)).sum()
            pct_both_zero = (both_zero_valid / n_valid_points) * 100
            
            xor_zero_valid = (
                (valid_data["HepG2"] == 0) ^ (valid_data["K562"] == 0)
            ).sum()
            pct_xor_zero = (xor_zero_valid / n_valid_points) * 100 
            
            xor_nan = (pivot_df["HepG2"].isna() ^ pivot_df["K562"].isna()).sum()
            pct_xor_nan = (xor_nan / n_total_points) * 100 
            
            both_nan = (pivot_df["HepG2"].isna() & pivot_df["K562"].isna()).sum()
            pct_both_nan = (both_nan / n_total_points) * 100 

            # Create scatterplot with centered axes
            fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

            # Plot points colored by Column Type
            sns.scatterplot(
                data=pivot_df,
                x="HepG2",
                y="K562",
                hue=plot_types[plot_type]['hue'],
                palette=plot_types[plot_type]['palette'],
                hue_order = plot_types[plot_type]['hue_order'],
                alpha=0.4,
                s=10,
                edgecolor='black',
                linewidth=0.6,
                ax=ax
            )
            ax.legend(loc='lower right', frameon=True, fontsize=12, markerscale=4)

            # Add text annotations for top 20 highest and lowest
            texts = []
            for idx in annotate_indices:
                
                col_name = pivot_df.loc[idx, "Column"]
                clean_name = col_name.replace("-interaction-shap", "").replace("-main-shap", "")
                
                if plot_type == "Main & Interaction":
                    if col_name.endswith("-main-shap"):
                        color = plot_types[plot_type]['palette']["Main"]
                    elif col_name.endswith("-interaction-shap"):
                        color = plot_types[plot_type]['palette']["Interaction"]
                
                elif plot_type == "Interaction Only":
                    is_ppi_result = pivot_df.loc[idx, "PPI Type"] == "PPI"
                    color = plot_types[plot_type]['palette']["PPI"] if is_ppi_result else plot_types[plot_type]['palette']["Not PPI"]

                text_obj = ax.text(
                    pivot_df.loc[idx, "HepG2"],
                    pivot_df.loc[idx, "K562"],
                    clean_name,
                    fontsize=5,
                    alpha=1,
                    color=color, 
                    zorder=10
                )
                texts.append(text_obj)
            
            adjustText.adjust_text(
                texts,
                arrowprops=dict(arrowstyle='-', lw=1, color='gray'),
                # force_points=.2,
                # force_text=.2,
                # expand_text=(.2, .2),   # Increase distance between text labels
                # expand_points=(.2, .2),  # Increase distance from points
            )

            # Move spines to center at (0, 0)
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Add diagonal reference line
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()]),
            ]
            ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0, linewidth=2)

            ax.set_aspect('equal')
            ax.grid(True, alpha=0.1,)

            ax.set_xlabel("")
            ax.set_ylabel("")
            # Label the axes using ax.text objects for custom positioning and style
            ax.text(
                0.6, -0.04,
                f"HepG2: {latex_symbol} (X-axis)",
                transform=ax.transAxes,
                fontsize=18,
                fontweight='bold',
                ha='center',
                va='top'
            )
            ax.text(
                -0.03, 0.55,
                f"K562: {latex_symbol} (Y-axis)",
                transform=ax.transAxes,
                fontsize=18,
                fontweight='bold',
                ha='right',
                va='center',
                rotation=90
            )

            # Add statistics text box
            stats_text = (
                f"Shared \"{plot_type}\" Points: {n_total_points:,}\n"
                f"Non-NaN Points (Shown): {n_valid_points:,}\n"
                f"Pearson: {pearson_r:.3f}\n"
                f"Spearman: {spearman_r:.3f}\n"
                f"% 0 shown (Both): {pct_both_zero:.0f}%\n"
                f"% 0 shown (XOR): {pct_xor_zero:.0f}%\n"
                f"% NaN (Both): {pct_both_nan:.0f}%\n"
                f"% NaN (XOR): {pct_xor_nan:.0f}%"
            )
            ax.text(
                0.05, 0.9,
                stats_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

            ax.set_title(
                f"{latex_symbol}: {plot_type} Comparison Between Cell Lines\n",
                fontsize=12,
                fontweight='bold'
            )

            fig.suptitle(
                f"{plot_notes}{plot_types[plot_type]['supplementary_note']}",
                fontsize=8,
                y=0.98,
                x=0.52
            )

            plt.tight_layout()
            plt.show()

    
    def create_top_cell_line_average_interactions_barplot(self): 
        val_col = f"Value - Signed-Local-SHAP-Mean-Bound-Only"
        latex_symbol = self.CONFIG["LATEX_SYMBOLS"]["Signed-Local-SHAP-Mean-Bound-Only"]

        # Load the table
        table = self.retrieve_shap_values_for_metric(metric=val_col.split(" - ")[-1])

        # Filter for interaction columns
        table = table.filter(
            (pl.col("Column Type") == "interaction") 
        )

        # Pivot to wide format to get averages across cell lines
        pivot_table = table.select(["Column", "Cell Line", val_col]).pivot(
            values=val_col,
            index="Column",
            columns="Cell Line", 
            sort_columns=True,
        ).sort("Column").drop_nans().drop_nulls().filter(
            (pl.col("HepG2") != 0) & (pl.col("K562") != 0)
        )

        pivot_df = pivot_table.to_pandas().reset_index()

        # Calculate average SHAP value across cell lines for ranking
        pivot_df["Average_Value"] = (pivot_df["HepG2"] + pivot_df["K562"]) / 2

        # Get the top 20 interactions by average value
        top_20_high_cols = pivot_df.nlargest(20, "Average_Value")["Column"].tolist()
        top_20_low_cols = pivot_df.nsmallest(20, "Average_Value")["Column"].tolist()
        top_40_cols = set(top_20_high_cols + top_20_low_cols)

        # Filter original table for these top columns
        plot_table = table.filter(
            pl.col("Column").is_in(top_40_cols)
        ).with_columns(
            pl.col("Column").str.replace("-interaction-shap", "").alias("Interaction Name")
        ).select(
            ["Interaction Name", "Cell Line", val_col]
        ).sort(val_col, descending=False)

        plot_df = plot_table.to_pandas()

        # Plotting
        fig = plt.figure(figsize=(10, 12), dpi=100)
        ax = sns.barplot(
            data=plot_df,
            x=val_col,
            y="Interaction Name",
            hue="Cell Line",
            orient="h",
            edgecolor='black',
            palette = {'K562': 'royalblue', 'HepG2': 'gold'},
            width=0.5
        )

        ax.axvline(0, color='black', linestyle='-', linewidth=2)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.tick_params(axis='y', which='both', left=False, labelsize=10)

        ax.set_title(f"Top Highest & Lowest Interactions by Average {latex_symbol} Across Cell Lines", pad=20, fontsize=18)

        ax.set_xlabel(f"{latex_symbol}", fontsize=24)
        ax.set_ylabel("Interaction", fontsize=24, labelpad=10, fontweight='bold')

        ax.legend(title="Cell Line", loc="best")

        ax.grid(axis='x', linestyle='--', alpha = 0.8)
        ax.grid(axis='y', linestyle='-', alpha=0.3)

        fig.suptitle(
            "NOTE: Features needed to have non-zero values in BOTH cell lines to be considered\n\n", 
            fontsize=14,
            y=0.95,
            x=0.52
        )

    
    def plot_interaction_sign_concordance_over_shap_thresholds(self): 
        val_col = f"Value - Signed-Local-SHAP-Mean-Bound-Only"
        latex_symbol = self.CONFIG["LATEX_SYMBOLS"]["Signed-Local-SHAP-Mean-Bound-Only"]

        # Load the table
        table = self.retrieve_shap_values_for_metric(metric=val_col.split(" - ")[-1])

        # Filter for interaction columns
        table = table.filter(
            (pl.col("Column Type") == "interaction") 
        )

        # Pivot to wide format: columns as rows, cell lines as columns
        # Filter to keep only rows where both cell lines have non-zero absolute values
        pivot_table = table.select(["Column", "Cell Line", val_col]).pivot(
            values=val_col,
            index="Column",
            columns="Cell Line",
            sort_columns=True,
        ).sort("Column").drop_nans().drop_nulls().filter(
            (pl.col("HepG2").abs() > 0) & (pl.col("K562").abs() > 0)
        )

        results = []
        for threshold in self.CONFIG["RAW_SHAP_THRESHOLDS"]: 
            # Subset the table based on the threshold
            filtered_table = pivot_table.filter(
                (pl.col("HepG2").abs() > threshold) & 
                (pl.col("K562").abs() > threshold)
            )

            # Calculate the percentage of rows with the same sign
            same_sign_count = ((filtered_table["HepG2"] > 0) & (filtered_table["K562"] > 0)).sum() + \
                               ((filtered_table["HepG2"] < 0) & (filtered_table["K562"] < 0)).sum()
            total_count = filtered_table.height

            percentage_same_sign = (same_sign_count / total_count) * 100

            # Append results to a new table
            results.append({
                "SHAP Threshold": threshold,
                "% Direction Concordance": percentage_same_sign, 
                "# Same Sign": same_sign_count,
                "# Total": total_count
            })

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results).sort_values(by="SHAP Threshold")

        with plt.style.context("../../../paper.mplstyle"): 
            # Create a scatter plot with color representing total interactions considered
            fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
            
            # Line plot to connect the dots (without color parameter)
            sns.lineplot(
                data=results_df,
                x="SHAP Threshold",
                y="% Direction Concordance",
                color="#E63030",
                linewidth=1.5,
                ax=ax,
                legend=False, 
                zorder=0
            )
            
            # Scatterplot with color parameter for legend using cividis palette
            scatter = ax.scatter(
                results_df["SHAP Threshold"],
                results_df["% Direction Concordance"],
                c=results_df["# Total"],
                cmap="YlGnBu",
                s=70,
                alpha=1.0,
                edgecolor="black",
                linewidth=1.2, 
                zorder=9
            )
            
            # Add colorbar for the # Total values
            cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
            cbar.set_label("# Interaction Features Considered", fontsize=8, labelpad=10)
            
            # Set log scale for x-axis
            ax.set_xscale('log')
            ax.grid(True, alpha=0.25, zorder=0, linestyle ='-.')

            ax.tick_params(axis='both', labelsize=12)

            ax.set_ylim(0, 105)        
            ax.set_title("% Both-Cell-Line-Learned Interactions with Same Direction as function of Min. SHAP Threshold", fontsize=7, pad=15)
            ax.set_xlabel(f"|{latex_symbol}| Threshold", fontsize=13, fontweight='bold', labelpad=-1)
            ax.set_ylabel("% Concordant Interactions", fontsize=12, fontweight='bold')

            fig.suptitle(
                (
                    "NOTE 1: Only considers interactions where BOTH cell lines have non-zero \"Signed Mean Average\" values\n"
                    "NOTE 2: Threshold means that BOTH cell lines must have |\"Signed Mean Average\"| > threshold to be included\n"
                    "\n"
                ), 
                fontsize=6,
                y=0.93, 
                x=0.48
            )

            plt.tight_layout()
            plt.savefig(self.CONFIG["FIGURES"]["cell_line_interaction_sign_similarity"], dpi=1000, bbox_inches='tight')
            plt.show()


    def plot_position_combination_directionality(self):
        metric = "Signed-Local-SHAP-Mean-Bound-Only"
        val_col = f"Value - {metric}"
        latex_symbol = self.CONFIG["LATEX_SYMBOLS"][metric]

        # Create 6x6 subplot grid for position pairs
        positions = self.CONFIG["POSITIONS"]
        n_pos = len(positions)

        # Load the table
        table = self.retrieve_shap_values_for_metric(metric=metric)

        # Filter for rows with non-zero floating point values
        table = table.filter(
            (pl.col("Column Type") == "interaction") &
            (~pl.col(val_col).is_null()) &
            (~pl.col(val_col).is_nan()) &
            (pl.col(val_col).abs() > 0)
        ).sort(
            ["Cell Line", "Column"]
        )
        
        # Common plot settings
        sharex = True
        sharey = True
        hspace = 0.05
        wspace = 0.05
        tick_fontsize = 6
        title_fontsize = 8
        tight_layout_rect = [0, 0, 1, 0.98]
        cell_line_palette = {
            'HepG2': 'tab:blue',
            'K562': 'tab:orange',
        }
        
        # Prepare data for each plot type
        # For barplot: use grouped_df with "Mean Value" column
        # For violinplot: use table converted to pandas with val_col
        table_pandas = table.select(["Cell Line", "Position 1", "Position 2", val_col]).to_pandas()
        
        # Group by cell line, position 1, position 2 and aggregate
        grouped = table.group_by(["Cell Line", "Position 1", "Position 2"]).agg([
            pl.col(val_col).mean().alias("Mean Value"),
            pl.col(val_col).count().alias("Count")
        ]).sort(["Cell Line", "Position 1", "Position 2"])
        # Convert grouped data to pandas for easier plotting
        grouped_df = grouped.to_pandas()
        
        plot_notes = f"NOTE: Input data was all rows per cell line with\nnon-zero (learned) {latex_symbol} values\n\n"

        plot_configs = [
            {
                "plot_type": "barplot",
                "y_col": "Mean Value",
                "data": grouped_df,
                "suptitle": f"Mean({latex_symbol}) by Position Pair & Cell Line",
                "supylabel": f"Mean({latex_symbol})",
                "supxlabel": "Cell Line",
            },
            {
                "plot_type": "violinplot",
                "y_col": val_col,
                "data": table_pandas,
                "suptitle": f"{latex_symbol} Distributions by Position Pair & Cell Line",
                "supylabel": f"{latex_symbol}",
                "supxlabel": "Cell Line",
            },
        ]

        for config in plot_configs:
            fig, axes = plt.subplots(
                nrows=n_pos, 
                ncols=n_pos, 
                figsize=(8,8), 
                dpi=400,
                sharex=sharex,
                sharey=sharey
            )
            
            plt.subplots_adjust(hspace=hspace, wspace=wspace)
            
            for row_idx, pos2 in enumerate(positions):
                for col_idx, pos1 in enumerate(positions):
                    ax = axes[row_idx, col_idx]
                
                    # Upper triangle (off-diagonal): turn off
                    if col_idx > row_idx:
                        ax.axis('off')
                        continue
                    
                    # Get subset data from the config's data source
                    data_df = config["data"]
                    subset = data_df[
                        (data_df["Position 1"] == pos1) & 
                        (data_df["Position 2"] == pos2)
                    ]
                    
                    assert not subset.empty, f"No data found for position pair Pos {pos1} - Pos {pos2}"
                    
                    # Create the appropriate plot
                    if config["plot_type"] == "barplot":
                        sns.barplot(
                            data=subset,
                            x="Cell Line",
                            y=config["y_col"],
                            hue="Cell Line",
                            width = 0.4,
                            palette=cell_line_palette,
                            order=self.CONFIG["CELL_LINES"],
                            hue_order=self.CONFIG["CELL_LINES"],
                            edgecolor='black',
                            linewidth=0.5,
                            ax=ax,
                            legend=False
                        )
                        
                        # Apply custom hatching and coloring to bars
                        for bar_idx, bar in enumerate(ax.patches):
                            bar_height = bar.get_height()
                            # Determine color based on value (positive = salmon, negative = skyblue)s
                            if bar_height >= 0:
                                bar.set_facecolor('salmon')
                            else:
                                bar.set_facecolor('skyblue')

                    elif config["plot_type"] == "violinplot":
                        sns.violinplot(
                            data=subset,
                            x="Cell Line",
                            y=config["y_col"],
                            hue="Cell Line",
                            palette=cell_line_palette,
                            order=self.CONFIG["CELL_LINES"],
                            hue_order=self.CONFIG["CELL_LINES"],
                            cut=0,
                            linewidth=0.5,
                            ax=ax,
                            legend=False
                        )
                    
                    # Add horizontal line at y = 0
                    ax.axhline(y=0, color='green', linestyle='--', linewidth=0.5, zorder=10)
                    
                    # Add position label in subplot title
                    ax.set_title(f"{pos1} & {pos2}", fontsize=title_fontsize, y=0.95)
                    
                    # Clean up spines
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    ax.tick_params(axis='both', labelsize=tick_fontsize)
                    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            
            # Always add row/column labels
            for row_idx, pos2 in enumerate(positions):
                axes[row_idx, 0].set_ylabel(f"Pos. {pos2}", fontsize=10, fontweight='bold')
            for col_idx, pos1 in enumerate(positions):
                if col_idx <= n_pos - 1:
                    axes[n_pos - 1, col_idx].set_xlabel(f"Pos. {pos1}", fontsize=10, fontweight='bold')
            
            # # Create legend
            # legend_handles = [
            #     mpatches.Patch(color=cell_line_palette['HepG2'], label='HepG2'),
            #     mpatches.Patch(color=cell_line_palette['K562'], label='K562'),
            #     plt.Line2D([0], [0], color='green', linestyle='--', linewidth=1, label='y=0')
            # ]
            # fig.legend(
            #     handles=legend_handles,
            #     loc='upper right',
            #     bbox_to_anchor=(0.98, 0.98),
            #     fontsize=8,
            #     frameon=True
            # )
            
            fig.suptitle(
                f"{plot_notes}{config['suptitle']}",
                fontsize=14,
                fontweight='bold',
                y=0.97, 
                color = 'goldenrod'
            )
            
            fig.supylabel(config["supylabel"], fontsize=20, x=0.01, color = 'goldenrod')
            fig.supxlabel(config["supxlabel"], fontsize=20, y=0.01, x=0.56, color = 'goldenrod')
            
            plt.tight_layout(rect=tight_layout_rect)
            plt.show()


    def retrieve_cell_line_concordant_interaction_values_for_rbp_prefix(self, rbp_prefix=None, position1=None, position2=None): 
        assert rbp_prefix is not None, "rbp_prefix must be provided."
        val_col = f"Value - Signed-Local-SHAP-Mean-Bound-Only"

        interactions = self.retrieve_shap_values_for_metric(metric=f'{val_col.split(" - ")[-1]}').filter(
            (pl.col("Sorted RBP Pair").str.contains(rbp_prefix)) &
            (pl.col("Column Type") == "interaction") 
        )

        if position1 is not None and position2 is not None: 
            assert position1 <= position2, "position1 must be less than or equal to position2."

            interactions = interactions.filter(
                (pl.col("Position 1") == position1) &
                (pl.col("Position 2") == position2)
            )
        
        # Pivot to wide format: columns as index, cell lines as separate columns for values and UBP counts
        pivot_table = interactions.select(["Column", "Cell Line", val_col, f"# UBPs - {val_col.split(' - ')[-1]}"]).pivot(
            values=[val_col, f"# UBPs - {val_col.split(' - ')[-1]}"],
            index="Column",
            columns="Cell Line",
            sort_columns=True,
        ).drop_nulls().drop_nans()

        # Filter to keep only rows where signs match between cell lines
        pivot_table = pivot_table.filter(
            ((pl.col(f"{val_col}_HepG2") > 0) & (pl.col(f"{val_col}_K562") > 0)) |
            ((pl.col(f"{val_col}_HepG2") < 0) & (pl.col(f"{val_col}_K562") < 0))
        ).with_columns(
            # Calculate average of absolute values across cell lines
            ((pl.col(f"{val_col}_HepG2").abs() + pl.col(f"{val_col}_K562").abs()) / 2).alias("avg_abs_value")
        ).sort("avg_abs_value", descending=True)

        return pivot_table


    def plot_psi_distributions_for_interaction_feature(self, interaction_feature=None): 
        assert interaction_feature.endswith("-interaction-shap"), "interaction_feature must end with '-interaction-shap'."

        # Get the two binding features from the interaction feature name
        binding_features = self.get_rbp_position_from_column(interaction_feature, binding_fmt=True)
        assert isinstance(binding_features, tuple) and len(binding_features) == 2, "Expected tuple of 2 binding features."
        
        feature_1_binding, feature_2_binding = binding_features

        # Extract RBP names and positions from binding features
        feature_1_clean = feature_1_binding.replace("_binding", "")
        feature_2_clean = feature_2_binding.replace("_binding", "")

        rbp_1, pos_1 = self.split_rbp_position(feature_1_clean)
        rbp_2, pos_2 = self.split_rbp_position(feature_2_clean)

        # Lazy load first order SHAP cache for both cell lines and verify schema
        all_plot_rows = []

        for cell_line in self.CONFIG["CELL_LINES"]:
            first_order_shap_lf = pl.scan_ipc(
                f"{self.CONFIG['FIRST_ORDER_SHAP_CACHE_DIR']}/{cell_line}_all-data.feather"
            )
            
            # Verify that required columns exist in schema
            schema = first_order_shap_lf.collect_schema()
            required_cols = {
                feature_1_binding, 
                feature_2_binding, 
                "Target_PSI",
                "RBP_KD_Target",
                f"has_RBP_KD_{pos_1}",
                f"has_RBP_KD_{pos_2}"
            }
            schema_cols = set(schema.names())
            assert required_cols.issubset(schema_cols), f"Missing columns in {cell_line}: {required_cols - schema_cols}"
            
            # Collect data once for this cell line
            cell_line_data = first_order_shap_lf.select(list(required_cols)).collect()
            
            # Define binding modes and their corresponding filter conditions
            binding_modes = [
                {
                    "name": "Both Bound",
                    "filter": (pl.col(feature_1_binding) == 1) & (pl.col(feature_2_binding) == 1)
                },
                {
                    "name": f"{feature_1_clean} | No {feature_2_clean}",
                    "filter": (pl.col(feature_1_binding) == 1) & (pl.col(feature_2_binding) == 0)
                },
                {
                    "name": f"{feature_2_clean} | No {feature_1_clean}",
                    "filter": (pl.col(feature_1_binding) == 0) & (pl.col(feature_2_binding) == 1)
                },
                {
                    "name": f"{feature_1_clean} RBP KD",
                    "filter": (pl.col("RBP_KD_Target") == rbp_1) & (pl.col(f"has_RBP_KD_{pos_1}") == True)
                },
                {
                    "name": f"{feature_2_clean} RBP KD",
                    "filter": (pl.col("RBP_KD_Target") == rbp_2) & (pl.col(f"has_RBP_KD_{pos_2}") == True)
                },
            ]
            
            # Process each binding mode
            for mode in binding_modes:
                filtered_data = cell_line_data.filter(mode["filter"])
                for psi in filtered_data["Target_PSI"].to_list():
                    all_plot_rows.append({
                        "Cell Line": cell_line,
                        "Binding Mode": mode["name"],
                        "Target_PSI": psi
                    })

        # Convert to polars DataFrame first, then to pandas for plotting
        plot_df = pl.DataFrame(all_plot_rows).to_pandas()

        # Define ordering by extracting names from binding_modes
        cell_line_order = self.CONFIG["CELL_LINES"]
        binding_mode_order = [mode["name"] for mode in binding_modes]

        # Create violinplot
        plt.figure(figsize=(8, 5), dpi=300)
        ax = sns.violinplot(
            data=plot_df,
            x="Binding Mode",
            y="Target_PSI",
            hue="Cell Line",
            hue_order=cell_line_order,
            order=binding_mode_order,
            cut=0,
            density_norm='width'
        )

        # Add counts and mean values above each violin
        max_psi = plot_df["Target_PSI"].max()
        y_text_height = max_psi * 1.03

        n_hues = len(cell_line_order)
        group_width = 0.8
        step = group_width / n_hues

        for i, binding_mode in enumerate(binding_mode_order):
            for j, cell_line in enumerate(cell_line_order):
                subset = plot_df[
                    (plot_df["Binding Mode"] == binding_mode) & 
                    (plot_df["Cell Line"] == cell_line)
                ]
                
                if len(subset) > 0:
                    n_points = len(subset)
                    mean_val = subset["Target_PSI"].mean()
                    
                    x_pos = i - (group_width / 2) + (step / 2) + j * step
                    
                    ax.text(
                        x_pos,
                        y_text_height,
                        f"{n_points:,}\n{mean_val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7
                    )

        ax.set_ylim(top=max_psi * 1.15)
        ax.tick_params(axis='x', labelsize=8)

        ax.set_xlabel("Binding Mode", fontsize=10, fontweight='bold')
        ax.set_ylabel("Target PSI", fontsize=10, fontweight='bold')
        ax.set_title(
            f"{interaction_feature.replace('-interaction-shap', '')}\nPSI Distributions by Binding Mode",
            y=1.02,
            fontsize=12
        )

        ax.legend(
            title="Cell Line",
            bbox_to_anchor=(1.02, 0.5),
            loc="center left",
            borderaxespad=0.0,
            frameon=True,
        )

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


    def plot_local_main_vs_interaction_val_for_interaction_cobinding(self, interaction=None): 
        assert interaction.endswith("-interaction-shap"), f"Expected interaction to end with '-interaction-shap', but got {interaction}"

        # Get the two main SHAP columns for the RBPs in this interaction
        binding_features = self.get_rbp_position_from_column(interaction, binding_fmt=False)
        rbp1_feature, rbp2_feature = binding_features

        # Construct main SHAP column names (e.g., "YBX3_3-main-shap", "YBX3_4-main-shap")
        main_shap_col1 = f"{rbp1_feature}-main-shap"
        main_shap_col2 = f"{rbp2_feature}-main-shap"
        
        dataframes = []
        for cell_line in self.CONFIG["CELL_LINES"]:
            ubp_ids = sorted(self.retrieve_UBP_ids_for_metric(cell_line = cell_line, metric="Signed-Local-SHAP-Mean-Bound-Only", column=interaction))
            
            ubp_id_table = pl.scan_csv(
                glob.glob(
                    f"{self.CONFIG['UBP_ID_DIR']}/{cell_line}*.tsv.gz"
                )[0],
                separator="\t",
            ).filter(
                pl.col(self.CONFIG["UBP_COL_NAME"]).is_in(ubp_ids)
            ).collect()

            model_hash= self.CONFIG["XGBOOST_BEST_MODEL_HASHES"][cell_line]
            model_path = f'{self.CONFIG["MODEL_DIR"]}/{model_hash}.pkl.gz'

            # Load the model
            with gzip.open(model_path, 'rb') as f:
                model = pickle.load(f)

            pred_input = ubp_id_table.select(
                model.column_order_when_fitting
            )

            predictions = model.predict(pred_input.to_pandas())
            ubp_id_table = ubp_id_table.with_columns(
                pl.Series("Predictions", predictions)
            )

            interaction_values_files = sorted(
                glob.glob(
                    f"{self.CONFIG['INTERACTION_VALUES_DIR']}/{cell_line}_*.feather"
                )
            )
            interaction_lf = pl.scan_ipc(interaction_values_files).filter(
                pl.col(self.CONFIG["UBP_COL_NAME"]).is_in(ubp_ids)
            )

            # Select the interaction SHAP column and the two main SHAP columns
            interaction_data = interaction_lf.select(
                [
                    self.CONFIG["UBP_COL_NAME"],
                    interaction,
                    main_shap_col1,
                    main_shap_col2
                ]
            ).collect()

            # Join ubp_id_table with interaction_data on UBP ID
            combined_data = ubp_id_table.select(
                [self.CONFIG["UBP_COL_NAME"], "Predictions"]
            ).join(
                interaction_data,
                on=self.CONFIG["UBP_COL_NAME"],
                how="left",
                validate="1:1"
            )

            combined_data = combined_data.with_columns(
                pl.lit(cell_line).alias("Cell Line")
            )
            dataframes.append(combined_data)

        final_df = pl.concat(dataframes)

        # Create figure with GridSpec for custom height ratios: top bar plot (small), 3D plots (medium), scatter plots (medium)
        fig = plt.figure(figsize=(12, 16), dpi=300)
        gs = GridSpec(3, 2, figure=fig, height_ratios=[0.3, 0.5, 0.2], hspace=0.02, wspace=0.2)
        cell_lines = self.CONFIG["CELL_LINES"]
        cell_line_data_dict = {}

        # --- Top row: Bar plot of feature SHAP values ---
        ax_bar = fig.add_subplot(gs[0, :])
        
        # Load SHAP values for the metric
        metric = "Signed-Local-SHAP-Mean-Bound-Only"
        val_col = f"Value - {metric}"
        latex_symbol = self.CONFIG["LATEX_SYMBOLS"][metric]
        shap_table = self.retrieve_shap_values_for_metric(metric=metric)
        
        # Prepare bar plot data for both cell lines
        bar_data = []
        feature_names = [rbp1_feature, rbp2_feature, f"{rbp1_feature} & {rbp2_feature}"]
        column_names = [main_shap_col1, main_shap_col2, interaction]
        
        for cell_line in cell_lines:
            for feature_name, column_name in zip(feature_names, column_names):
                subset = shap_table.filter(
                    (pl.col("Cell Line") == cell_line) &
                    (pl.col("Column") == column_name)
                )
                assert subset.height == 1, f"Expected exactly one row for {cell_line} and {column_name}, but got {subset.height}"

                value = subset[val_col].item()
                bar_data.append({
                    "Feature": feature_name,
                    "Value": value,
                    "Cell Line": cell_line
                })
        
        bar_df = pd.DataFrame(bar_data)
        # Define palette for cell lines
        cell_line_palette = {'HepG2': 'steelblue', 'K562': 'coral'}
        
        # Use seaborn barplot with better control
        sns.barplot(
            data=bar_df,
            x="Feature",
            y="Value",
            hue="Cell Line",
            order=feature_names,
            hue_order=cell_lines,
            ax=ax_bar,
            palette=cell_line_palette,
            edgecolor='black',
            linewidth=1.2,
            width=0.4,
            gap=0.2,
            legend=True
        )
        
        ax_bar.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax_bar.set_xlabel("Main/Interaction Feature", fontsize=12)
        ax_bar.set_ylabel(latex_symbol, fontsize=18)
        ax_bar.set_title(f"What motivated this analysis -- strong, opposing main vs interaction {latex_symbol}", fontsize=14)
        ax_bar.grid(True, alpha=0.3, axis='y')

        # Add value labels on top of each bar
        for patch in ax_bar.patches:
            height = patch.get_height()

            # Skip patches with zero or near-zero height (e.g., the axhline y=0)
            if abs(height) < 1e-10:
                continue
            
            ax_bar.text(
                patch.get_x() + patch.get_width() / 2.,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom' if height >= 0 else 'top',
                fontsize=10,
                fontweight='bold'
            )
        
        # Position legend outside right of plot
        ax_bar.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14, title="Cell Line", frameon=True, title_fontsize=16)

        for subplot_idx, cell_line in enumerate(cell_lines, 1):
            ax = fig.add_subplot(gs[1, subplot_idx - 1], projection='3d')
            
            # Filter data for this cell line
            cell_line_data = final_df.filter(pl.col("Cell Line") == cell_line).to_pandas()
            cell_line_data_dict[cell_line] = cell_line_data
            
            # Define the three SHAP features
            shap_features = [
                main_shap_col1,
                main_shap_col2,
                interaction
            ]
            feature_labels = [rbp1_feature, rbp2_feature, f"{rbp1_feature} & {rbp2_feature}"]
            x_positions = np.arange(len(shap_features))
            
            # Set labels and title first
            ax.set_xlabel("Main/Interaction Feature", fontsize=10, labelpad=10)
            ax.set_ylabel("Model Prediction", fontsize=10, labelpad=10)
            ax.set_zlabel("Co-binding unique binding pattern's Local SHAP", fontsize=8, labelpad=10, fontweight='bold')
            
            ax.set_title(f"{cell_line}\nONLY co-binding unique binding patterns", fontsize=14, y=0.95)
            
            # Set x-axis ticks
            ax.set_xticks(x_positions)
            ax.set_xticklabels(feature_labels, fontsize=8, color='red', zorder=10)
            
            # Adjust viewing angle for better visibility BEFORE plotting data
            ax.view_init(elev=15, azim=45)
            
            # Get z-axis limits before plotting to properly scale shadows
            z_min = min([row[main_shap_col1] for _, row in cell_line_data.iterrows()] + 
                       [row[main_shap_col2] for _, row in cell_line_data.iterrows()] + 
                       [row[interaction] for _, row in cell_line_data.iterrows()])
            z_max = max([row[main_shap_col1] for _, row in cell_line_data.iterrows()] + 
                       [row[main_shap_col2] for _, row in cell_line_data.iterrows()] + 
                       [row[interaction] for _, row in cell_line_data.iterrows()])
            shadow_z = z_min - (z_max - z_min) * 0.1  # Place shadow slightly below minimum
            
            # # Add reference plane at z=0 (local SHAP = 0)
            # xx, yy = np.meshgrid(
            #     np.linspace(-0.5, len(shap_features) - 0.5, 10),
            #     np.linspace(cell_line_data["Predictions"].min(), cell_line_data["Predictions"].max(), 10)
            # )
            # zz = np.zeros_like(xx)
            # ax.plot_surface(xx, yy, zz, alpha=0.1, color='red', label='z=0 reference plane')
            
            # For each UBP, create a line connecting the three SHAP values
            for _, row in cell_line_data.iterrows():
                z_values = [row[main_shap_col1], row[main_shap_col2], row[interaction]]
                y_value = row["Predictions"]
                
                # Plot the line for this UBP (x=feature type, y=predictions, z=SHAP value)
                ax.plot(
                    x_positions, 
                    [y_value] * len(shap_features), 
                    z_values,
                    alpha=0.3,
                    linewidth=0.45,
                    color='skyblue',
                    marker='o', 
                    markerfacecolor='black',
                    markeredgewidth=0.5,
                    markersize=3
                )
                
                # Add shadow projection on the ground plane (all z-values at shadow_z level)
                ax.plot(
                    x_positions,
                    [y_value] * len(shap_features),
                    [shadow_z] * len(shap_features),
                    alpha=0.2,
                    linewidth=0.5,
                    color='darkgray',
                    linestyle='--'
                )
            
            # Add grid
            ax.grid(True, alpha=0.2)

        # Add scatter plots in bottom row: Model Prediction (x-axis) vs Interaction SHAP Value (y-axis)
        for subplot_idx, cell_line in enumerate(cell_lines, 1):
            ax_scatter = fig.add_subplot(gs[2, subplot_idx - 1])
            cell_line_data = cell_line_data_dict[cell_line]
            
            # Create scatter plot
            ax_scatter.scatter(
                cell_line_data["Predictions"],
                cell_line_data[interaction],
                alpha=0.25,
                s=5,
                edgecolor='black',
                linewidth=0.3,
                color='teal'
            )
        
            # Calculate correlations and number of points
            n_points = len(cell_line_data)
            pearson_r, _ = pearsonr(cell_line_data["Predictions"], cell_line_data[interaction])
            spearman_r, _ = spearmanr(cell_line_data["Predictions"], cell_line_data[interaction])
            
            ax_scatter.set_xlabel("Model Prediction", fontsize=12)
            ax_scatter.set_ylabel("LOCAL Interaction SHAP Value", fontsize=12)

            ax_scatter.set_title(f"{cell_line}\nONLY co-binding unique binding patterns", fontsize=14, y=1.02)
            
            ax_scatter.grid(True, alpha=0.2)
            
            # Add statistics text box in bottom left corner
            stats_text = (
                f"# Co-binding Unique Binding Patterns: {n_points:,}\n"
                f"Pearson: {pearson_r:.2f}\n"
                f"Spearman: {spearman_r:.2f}"
            )
            ax_scatter.text(
                0.05, 0.05,
                stats_text,
                transform=ax_scatter.transAxes,
                fontsize=8,
                verticalalignment='bottom',
            )

        plt.suptitle(
            f"Main vs Interaction LOCAL SHAP Values by Model Prediction:\n{interaction.replace('-interaction-shap', '')}",
            fontsize=14,
            y=0.95
        )

        plt.tight_layout()
        plt.show()


    def validate_selected_interactions_actual_psi_and_local_SHAP_distributions(self): 
        selected_interactions = [
            "YBX3_3-YBX3_4-interaction-shap", 
            "IGF2BP1_3-IGF2BP1_4-interaction-shap",
            "DDX3X_3-AQR_4-interaction-shap", 
            "YBX3_3-PRPF8_4-interaction-shap", 
            "DDX3X_3-SND1_4-interaction-shap", 
            "BUD13_3-BUD13_4-interaction-shap",
            "AQR_3-DDX3X_4-interaction-shap",
            "PTBP1_3-U2AF2_4-interaction-shap", 
            "DGCR8_3-AQR_4-interaction-shap",
            "SND1_3-PRPF8_4-interaction-shap",
        ]

        for interaction in selected_interactions:
            self.plot_psi_distributions_for_interaction_feature(interaction_feature=interaction)
            self.plot_local_main_vs_interaction_val_for_interaction_cobinding(interaction=interaction)    

    
    def create_table_from_systematic_screen_of_interactions_for_psi_changes(self): 

        OUTPUT_FILE = self.CONFIG["INTERACTION_PSI_CHANGES_SCREENING"]

        if pathlib.Path(OUTPUT_FILE).exists():
            logger.success("FROM CACHE: loading screened interaction PSI changes table")
            return pl.read_csv(OUTPUT_FILE, separator="\t")

        else: 
            logger.info("No cached table found, creating table for screened interaction PSI changes... ")

            metric = "Signed-Local-SHAP-Mean-Bound-Only"
            val_col = f"Value - {metric}"

            # Load signed SHAP metric
            raw_table = self.retrieve_shap_values_for_metric(metric=metric)

            # Subset to interaction rows with non-null, non-nan, > 0 values
            table = raw_table.filter(
                (pl.col("Column Type") == "interaction") &
                (~pl.col(val_col).is_null()) &
                (~pl.col(val_col).is_nan()) &
                (pl.col(val_col).abs() > 0)
            )

            """
            Create pivot table of interactions
            """
            # Pivot to wide format: columns as rows, cell lines as columns
            pivot_table = table.select(["Column", "Cell Line", val_col]).pivot(
                values=val_col,
                index="Column",
                columns="Cell Line",
                sort_columns=True,
            ).drop_nulls()

            # Filter to keep only rows where signs match between cell lines
            pivot_table = pivot_table.filter(
                ((pl.col("HepG2") > 0) & (pl.col("K562") > 0)) |
                ((pl.col("HepG2") < 0) & (pl.col("K562") < 0))
            ).sort("Column")


            # Add " Interaction" suffix to cell line columns
            pivot_table = pivot_table.rename({
                "Column": "Feature-Feature Interaction",
                "HepG2": "HepG2 - Interaction SHAP",
                "K562": "K562 - Interaction SHAP", 
            })

            """
            Add main effect SHAP values for the two features in each interaction
            """
            # For each interaction column, retrieve the corresponding main effect SHAP values
            results = []
            for interaction_col in pivot_table["Feature-Feature Interaction"].to_list():
                # Get the two main SHAP columns from the interaction column
                binding_features = self.get_rbp_position_from_column(interaction_col, binding_fmt=False)
                main_shap_col1 = f"{binding_features[0]}-main-shap"
                main_shap_col2 = f"{binding_features[1]}-main-shap"
                
                # Get the main effect values for both cell lines
                main_values = raw_table.filter(
                    pl.col("Column").is_in([main_shap_col1, main_shap_col2])
                ).select(["Column", "Cell Line", val_col])

                assert main_values.height == 4, f"Expected 4 rows for main effects of {interaction_col}, but got {main_values.height}"
                
                main_pivot = main_values.pivot(
                    values=val_col,
                    index="Column",
                    columns="Cell Line",
                    sort_columns=True,
                )
                            
                # Get the interaction row
                interaction_row = pivot_table.filter(pl.col("Feature-Feature Interaction") == interaction_col)
                assert interaction_row.height ==1, f"Expected exactly 1 row for interaction {interaction_col} in pivot table, but got {interaction_row.height}"
                
                row_dict = interaction_row.to_dicts()[0]
                
                # Add main effect values
                skip_row = False
                for feature_idx, main_shap_col in enumerate([main_shap_col1, main_shap_col2], start=1):
                    main_data = main_pivot.filter(pl.col("Column") == main_shap_col).to_dicts()[0]
                    hepg2_val = main_data["HepG2"]
                    k562_val = main_data["K562"]
                    
                    # Skip row if the two cell line values have different signs
                    if (hepg2_val > 0 and k562_val < 0) or (hepg2_val < 0 and k562_val > 0):
                        skip_row = True
                        break
                    
                    row_dict[f"HepG2 - F{feature_idx} ({feature_idx}{'st' if feature_idx==1 else 'nd'} Individual Feature in Interaction) SHAP"] = hepg2_val
                    row_dict[f"K562 - F{feature_idx} ({feature_idx}{'st' if feature_idx==1 else 'nd'} Individual Feature in Interaction) SHAP"] = k562_val
            
                if not skip_row:
                    results.append(row_dict)

            # Convert results to a polars DataFrame
            results_df = pl.DataFrame(results).sort("Feature-Feature Interaction")

            """
            Get First Order SHAP Cache tables with actual PSI distributions
            """
            first_order_shap_data = {}
            for cell_line in self.CONFIG["CELL_LINES"]:
                first_order_shap_lf = pl.scan_ipc(
                    f"{self.CONFIG['FIRST_ORDER_SHAP_CACHE_DIR']}/{cell_line}_all-data.feather"
                )
                
                # Verify that required columns exist in schema
                cols = first_order_shap_lf.collect_schema().names()
                required_cols = ["Target_PSI"] + [col for col in cols if col.endswith("_binding")]
                first_order_shap_data[cell_line] = first_order_shap_lf.select(required_cols).collect()

            """
            For each interaction, get summary statistics and run statistical tests across both cell lines
            """
            # For each interaction in results_df, calculate PSI distribution statistics
            interaction_stats = []
            distribution_names = ["Both Bound", "F1 Bound Only", "F2 Bound Only",] # "Either Bound"]
            metric_names = ["# Points", "Average", "Median"]
            
            for row in tqdm(results_df.iter_rows(named=True), desc="Processing interactions"):
                interaction_col = row["Feature-Feature Interaction"]
                
                # Get the two binding features from the interaction column
                binding_features = self.get_rbp_position_from_column(interaction_col, binding_fmt=True)
                feature_1_binding, feature_2_binding = binding_features

                row_dict = {
                    "Feature-Feature Interaction": interaction_col
                }
                
                # Process each cell line
                temp_data = {}
                distributions_by_cell_line = {}
                for cell_line in self.CONFIG["CELL_LINES"]:
                    cell_line_data = first_order_shap_data[cell_line]
                    # Create the three distributions based on binding patterns
                    both_bound = cell_line_data.filter(
                        (pl.col(feature_1_binding) == 1) & (pl.col(feature_2_binding) == 1)
                    )["Target_PSI"].to_list()
                    
                    f1_only = cell_line_data.filter(
                        (pl.col(feature_1_binding) == 1) & (pl.col(feature_2_binding) == 0)
                    )["Target_PSI"].to_list()
                    
                    f2_only = cell_line_data.filter(
                        (pl.col(feature_1_binding) == 0) & (pl.col(feature_2_binding) == 1)
                    )["Target_PSI"].to_list()
                    
                    # # Combine XOR distributions (f1 only + f2 only)
                    # xor_combined = f1_only + f2_only
                    
                    # Calculate statistics for each distribution
                    distributions = {
                        "Both Bound": both_bound,
                        "F1 Bound Only": f1_only,
                        "F2 Bound Only": f2_only,
                        # "Either Bound": xor_combined
                    }
                    
                    temp_data[cell_line] = {}
                    distributions_by_cell_line[cell_line] = distributions
                    for dist_name, dist_values in distributions.items():
                        n_points = len(dist_values)
                        avg_val = float(np.mean(dist_values))
                        median_val = float(np.median(dist_values)) 
                        
                        temp_data[cell_line][dist_name] = {
                            "# Points": n_points,
                            "Average": avg_val,
                            "Median": median_val
                        }
                
                # Build row_dict with sorted column order: distribution → metric → cell_line
                for dist_name in distribution_names:
                    for metric in metric_names:
                        for cell_line in self.CONFIG["CELL_LINES"]:
                            row_dict[f"{cell_line} - {dist_name}: {metric}"] = temp_data[cell_line][dist_name][metric]
                
                # Determine testing direction based on interaction SHAP sign
                hepg2_interaction_shap = row["HepG2 - Interaction SHAP"]
                k562_interaction_shap = row["K562 - Interaction SHAP"]

                assert (hepg2_interaction_shap > 0 and k562_interaction_shap > 0) or (hepg2_interaction_shap < 0 and k562_interaction_shap < 0), "Expected interaction SHAP values to have the same sign between cell lines due to earlier filtering, but got different signs."
                
                # Add testing direction column (assume both have same sign since we filtered earlier)
                testing_direction = "greater" if hepg2_interaction_shap > 0 else "less"
                row_dict["Statistical Testing Directionality"] = testing_direction
                
                # Add statistical test columns: Cell Line, Groups Compared, Statistical Test
                group_pairs = [
                    ("Both Bound", "F1 Bound Only"),
                    ("Both Bound", "F2 Bound Only")
                ]
                statistical_tests = ["Mann-Whitney U", "Welch's t-test"]
                
                for group1, group2 in group_pairs:
                    for test_name in statistical_tests:
                        for cell_line in self.CONFIG["CELL_LINES"]:
                            col_name = f"{cell_line} - {group1} vs {group2}: {test_name}"
                            
                            dist1 = distributions_by_cell_line[cell_line][group1]
                            dist2 = distributions_by_cell_line[cell_line][group2]
                            
                            if test_name == "Mann-Whitney U":
                                p_value = mannwhitneyu(dist1, dist2, alternative=testing_direction)[1]
                            elif test_name == "Welch's t-test":
                                p_value = ttest_ind(dist1, dist2, equal_var=False, alternative=testing_direction)[1]
                            
                            row_dict[col_name] = p_value
                
                interaction_stats.append(row_dict)

            # Convert to polars DataFrame and join with results_df
            stats_df = pl.DataFrame(interaction_stats)
            
            results_df = results_df.join(
                stats_df,
                on="Feature-Feature Interaction",
                how="left"
            )
            # Validate the results DataFrame
            assert results_df.null_count().sum_horizontal().item() == 0, "Null values found in results DataFrame"
            assert results_df.select(pl.col(pl.Float64).is_nan().sum()).sum_horizontal().item() == 0, "NaN values found in results DataFrame"
            
            results_df.write_csv(OUTPUT_FILE, separator="\t")
            logger.success(f"Screened interaction PSI changes table created and saved to {OUTPUT_FILE}")
            
            return results_df



#########################################################
############## NON-CLASS FUNCTIONS ######################
#########################################################

def return_parallelization_start_stop(columns, config): 
    cols_per_job = config["COLS_PER_JOB"]
    num_columns = len(columns)
    
    chunks = []
    for i in range(0, num_columns, cols_per_job):
        start = i + 1  # 1-indexed, inclusive
        stop = min(i + cols_per_job, num_columns)  # inclusive
        chunks.append((start, stop))

    expected_num_chunks = (num_columns // cols_per_job) + (1 if num_columns % cols_per_job != 0 else 0)
    assert len(chunks) == expected_num_chunks, f"Expected {expected_num_chunks} chunks, got {len(chunks)}"

    return chunks


def is_file_cached(metric, cell_line, start, stop, config):
    output_filename = f"{config['TMP_CACHE_DIR']}/{metric}_{cell_line}_{start}_{stop}.json"
    return pathlib.Path(output_filename).exists()


if __name__ == "__main__": 

    with open('./1_variable_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    parser = argparse.ArgumentParser(description="Second Order SHAP Analysis")
    parser.add_argument("--cell_line", choices=config["CELL_LINES"], help="Cell line to analyze")
    parser.add_argument("--metric", choices=config["VALID_FEATURE_METRICS"], help="Metric to calculate")
    parser.add_argument("--start", type=int, help="Start column (1-indexed, inclusive)")
    parser.add_argument("--stop", type=int, help="Stop column (1-indexed, inclusive)")
    parser.add_argument(
        "--parallelize_metric_calc",
        choices=config["VALID_FEATURE_METRICS"],
        help="Metric to parallelize over (must be a valid metric)"
    )
    parser.add_argument(
        "--parallelize_all_metric_calcs", 
        action="store_true",
        help="If set, will parallelize metric calculations for all metrics and cell lines."
    )
    parser.add_argument(
        "--aggregate_all_metrics", 
        action="store_true",
        help="If set, will aggregate all metrics after parallelized calculations are done."
    )

    args = parser.parse_args()

    def parallelize_metric_calculation(config, args): 
        analyzer = SecondOrderShapNetworkAnalyzer()

        for cell_line in config["CELL_LINES"]:
            features = analyzer.rbp_feature_metadata[cell_line]["Features"]
            chunks = return_parallelization_start_stop(features, config)

            counter = 0
            for start, stop in chunks:
                
                if not is_file_cached(args.parallelize_metric_calc, cell_line, start, stop, config):
                    sbatch_prefixes = config["SBATCH_PREFIXES"]

                    standard_to_parallel_ratio = 2  # 2:1 ratio for 66%/33%
                    # Use standard_to_parallel_ratio to determine prefix: 2 out of 3 times use [0], 1 out of 3 times use [1]
                    prefix = sbatch_prefixes[0] if (counter % (standard_to_parallel_ratio + 1)) < standard_to_parallel_ratio else sbatch_prefixes[1]
                    
                    cmd = (
                        f"{prefix} --output='../SLURM_logs/{args.parallelize_metric_calc}_{cell_line}_{start}_{stop}.out' "
                        f"--error='../SLURM_logs/{args.parallelize_metric_calc}_{cell_line}_{start}_{stop}.err' "
                        f"--wrap \"python3.11 {__file__} "
                        f"--cell_line {cell_line} "
                        f"--metric {args.parallelize_metric_calc} "
                        f"--start {start} "
                        f"--stop {stop}\""
                    )
                    os.system(cmd)
                    counter += 1

    if args.parallelize_metric_calc:
        parallelize_metric_calculation(config, args)
    
    elif args.parallelize_all_metric_calcs:
        
        for metric in config["VALID_FEATURE_METRICS"]:
            args.parallelize_metric_calc = metric
            parallelize_metric_calculation(config, args)
    
    elif args.aggregate_all_metrics:
        analyzer = SecondOrderShapNetworkAnalyzer()
        
        for metric in config["VALID_FEATURE_METRICS"]:
            analyzer.retrieve_shap_values_for_metric(metric=metric)
    
    elif args.metric and args.cell_line and args.start and args.stop:
        
        analyzer = SecondOrderShapNetworkAnalyzer()
        analyzer.calculate_metrics_for_column_range(
            cell_line=args.cell_line,
            metric=args.metric,
            start=args.start,
            stop=args.stop
        )
