import yaml, pathlib, glob, json, argparse, os, sys, copy

import pandas as pd, polars as pl, seaborn as sns, numpy as np, matplotlib.pyplot as plt, matplotlib.patches as mpatches

from dataclasses import dataclass
from loguru import logger
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from tqdm import tqdm
from itertools import combinations
from scipy.stats import pearsonr, spearmanr
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score


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
                if col.startswith("# UBPs - "):
                    table = table.with_columns(pl.col(col).cast(pl.UInt32))
            
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
                pl.col("RBP 2").str.to_uppercase()
            ])

            # Assert that Position 1 is always <= Position 2
            if not (df["Position 1"] <= df["Position 2"]).all():
                raise AssertionError("Found rows where Position 1 > Position 2.")

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
        assert opts["highlight"] in self.CONFIG["PPI_TYPES"] or opts["highlight"] is None, f"highlight type '{opts["highlight"]}' not recognized."

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
            f"{self.CONFIG["FIGURES"]["side_by_side_heatmap_dir"]}/{cell_line}_{ppi_source_str}_{metric}_highlight_{highlight_str}_side_by_side_heatmap.png",
            bbox_inches='tight',
            dpi=600
        )
        plt.show()

    
    def plot_roc_and_prc_curves_for_metric(self, metric = None): 
        assert metric in self.CONFIG["VALID_FEATURE_METRICS"], f"Metric '{metric}' not recognized."
        
        # table = self.retrieve_shap_values_for_metric(metric=metric)
        # # no main effect columns or different position but same RBP columns since those are not distinct-RBP PPIs
        # table = table.filter(
        #     (pl.col("Column Type") == "interaction")
        #     & (pl.col("RBP 1") != pl.col("RBP 2"))
        # )

        # table = self.add_ppi_stats_to_long_df(df = table)

        #TODO remove later 
        # table.write_csv("./tmp.tsv", separator="\t")
        table = pl.read_csv("./tmp.tsv", separator="\t")
        
        union_col = "Rec-Y2H/Street et al. IP-MS (Union)"
        # Build union column as a list
        union_col_values = []
        for rec_val, ip_val in zip(table["Rec-Y2H"].to_list(), table["Street et al | IP-MS"].to_list()):
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

        # Add union column to table
        table = table.with_columns(
            pl.Series(union_col, union_col_values)
        )

        ppi_sources = self.CONFIG["PPI_SOURCES"] + [union_col]
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
                                f"{self.CONFIG["FIGURES"]["prc_roc_curves_dir"]}/curve_inputs/{plot_type}_{cell_line}_{ppi_source_str_conversion}_{ppi_type}_curve_input_data.tsv",
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


    def tmp(self): 
        # for cell_line in ["K562"]: 
            
        #     table = self.retrieve_shap_values_for_metric(metric="Signed-Local-SHAP-Mean-Bound-Only")

        #     pivot_tables = self.convert_long_metric_table_to_symmetric_matrix(
        #         df=table.filter(pl.col("Cell Line") == cell_line),
        #         metric="Signed-Local-SHAP-Mean-Bound-Only"
        #     )

        pass








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
