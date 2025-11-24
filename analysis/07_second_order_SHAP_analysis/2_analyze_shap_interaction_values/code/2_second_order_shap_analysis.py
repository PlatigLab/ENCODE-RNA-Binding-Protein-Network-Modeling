import yaml, pathlib, glob, json, argparse, os, sys

import pandas as pd, polars as pl

from dataclasses import dataclass
from loguru import logger
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from tqdm import tqdm


@dataclass
class SecondOrderShapNetworkAnalyzer:

    YAML_CONFIG_FILE = "./1_variable_config.yaml"

    def __post_init__(self):
        
        with open(self.YAML_CONFIG_FILE, "r") as file:
            self.CONFIG = yaml.safe_load(file)

        self.load_features()
        #TODO uncomment later
        # self.load_ppi()

        logger.info("Class initialized and loaded.")


    def load_features(self):
        logger.info(f"Loading feature & RBP info per cell line... ")

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
        logger.success("SUCCESS: Feature & RBP info loaded.")


    def load_ppi(self):
        
        OUTPUT_FILE = self.CONFIG["PPI_INFO"]["cached_ppi_file"]

        if pathlib.Path(OUTPUT_FILE).exists(): 
            raise NotImplementedError("Loading cached PPI file not yet implemented.")
        
        else:
            logger.info("Cached PPI file not found. Generating PPI table for both Rec-Y2H and Street et al. Molecular Cell 2024 datasets...")

            self.create_protein_synonym_lookup_table(mode="recy2h")
            # Read in Rec-Y2H PPI file
            recy2h_ppi_table = self.load_rec_y2h_ppi()

            #TODO impement Street et al. later

            final_ppi_table = recy2h_ppi_table
            # self.create_protein_synonym_lookup_table(mode="street_et_al")
            # street_et_al_ppi_table = self.load_street_et_al_ppi()

            # final_ppi_table = pd.merge(
            #     recy2h_ppi_table,
            #     street_et_al_ppi_table,
            #     on=["Cell Line", "Interaction"],
            #     how="outer"
            # )

            final_ppi_table.to_csv(
                OUTPUT_FILE,
                sep="\t",
                index=False
            )
            logger.success("SUCCESS: PPI table generated and saved.")


    def create_protein_synonym_lookup_table(self, mode=None):
        assert mode in ["recy2h", "street_et_al",], "Mode must be one of 'recy2h' or 'street_et_al'."

        logger.info(f"Creating protein synonym lookup table for '{mode}' ...")

        # Uniprot mapping file 
        uniprot_mapping = self.CONFIG["PPI_INFO"]["recy2h_ppi_dir"]["uniprot_mapping"]
        
        if mode == "recy2h":
            screen_results_path = self.CONFIG["PPI_INFO"]["recy2h_ppi_dir"]["results"]
            all_rbps_screened_path = self.CONFIG["PPI_INFO"]["recy2h_ppi_dir"]["all_rbps_screened"]

        elif mode == "street_et_al":
            screen_results_path = self.CONFIG["PPI_INFO"]["street_et_al"]["table_s2_full_results"]
            all_rbps_screened_path = self.CONFIG["PPI_INFO"]["street_et_al"]["all_rbps_screened"]

        # mapping uniprot ids to gene names/synonyms
        uniprot_df = pd.read_csv(uniprot_mapping, sep="\t")
        uniprot_df["Gene name"] = uniprot_df["Gene name"].str.lower()
        uniprot_df["Gene Synonym"] = uniprot_df["Gene Synonym"].str.lower()

        if mode == "recy2h":
            # load all rec-Y2H screening results
            screening_results = pd.read_excel(screen_results_path)
            screening_results["Protein A"] = screening_results["Protein A"].str.lower()
            screening_results["Protein B"] = screening_results["Protein B"].str.lower()

            # Create RBP sets per cell line
            rbp_sets = {cell_line: set([rbp.lower() for rbp in self.rbp_feature_metadata[cell_line]["RBPs"]]) for cell_line in self.CONFIG["CELL_LINES"]}

            # Concatenate Protein A/B and UniProt accessions A/B, then create mapping
            protein_a = screening_results[["Protein A", "UniProt accessions A"]].rename(
                columns={"Protein A": "Protein", "UniProt accessions A": "Accession"}
            )
            protein_b = screening_results[["Protein B", "UniProt accessions B"]].rename(
                columns={"Protein B": "Protein", "UniProt accessions B": "Accession"}
            )
            
            protein_names = pd.concat([protein_a, protein_b], ignore_index=True)
            protein_names = protein_names.drop_duplicates(subset=["Protein", "Accession"])
            # Assert no duplicate values in the "Protein" column
            assert protein_names["Protein"].is_unique, "Duplicate values found in the 'Protein' column."

            all_rbps_screened = pd.read_excel(all_rbps_screened_path)
            all_rbps_screened = all_rbps_screened[["Gene Symbol"]]
            all_rbps_screened["Gene Symbol"] = all_rbps_screened["Gene Symbol"].str.lower()
            
            all_rbps_screened = all_rbps_screened.merge(protein_names, left_on="Gene Symbol", right_on="Protein", how="left")

            # Assert that there are no cases where only one of 'Protein' or 'Accession' is missing
            mask_protein = all_rbps_screened["Protein"].notnull()
            mask_accession = all_rbps_screened["Accession"].notnull()
            assert ((mask_protein == mask_accession).all()), "Rows found where only one of 'Protein' or 'Accession' is missing."

            # Find gene symbols that do not show up in the screening results but were screened
            missing_genes = all_rbps_screened[all_rbps_screened["Accession"].isnull()]["Gene Symbol"].tolist()
            logger.warning(f"{len(missing_genes)} gene symbols were screened but do not show up in the screening results: {missing_genes}")

            # key mapping: protein name -> uniprot accession
            protein_to_accession = dict(
                zip(protein_names["Protein"], protein_names["Accession"])
            )
        
        elif mode == "street_et_al":
            raise NotImplementedError("Street et al. mode not yet implemented.")
            all_rbps_screened_df = pd.read_excel(
                all_rbps_screened_path, 
                sheet_name="All baits", 
                header=1
            )
            return all_rbps_screened_df
        

        # Build lookup table to relate RBP names they mention to the RBP names we have for our eCLIP
        protein_synonym_lookup = {cell_line: {} for cell_line in self.CONFIG["CELL_LINES"]}
        # synonyms to manually validate later
        validate_synonyms = { cell_line: {} for cell_line in self.CONFIG["CELL_LINES"]}

        if mode == "recy2h":
            for protein, accession in protein_to_accession.items():
                for cell_line, rbp_set in rbp_sets.items():

                    # Direct match
                    if protein in rbp_set:
                        protein_synonym_lookup[cell_line][protein] = protein
                        
                    else: 
                        # Subset uniprot_df for possible synonyms
                        subset_df = uniprot_df[
                            (uniprot_df["Gene name"] == protein) |
                            (uniprot_df["Gene Synonym"] == protein) |
                            (uniprot_df["UniProtKB Gene Name ID"] == accession)
                        ]
                        
                        # no other possible names means that we do not have that RBP eCLIP'd
                        if subset_df.empty:
                            protein_synonym_lookup[cell_line][protein] = None
                        
                        else:
                            # Compile all possible names
                            all_names = set(
                                subset_df["Gene name"].dropna().str.lower().tolist() +
                                subset_df["Gene Synonym"].dropna().str.lower().tolist()
                            )

                            # Find intersection with RBP set
                            matches = rbp_set & all_names
                            assert len(matches) <= 1, "Expected at most one match per protein."

                            # Assign match if found
                            if len(matches) == 1:
                                validate_synonyms[cell_line][protein] = matches.pop()
                                protein_synonym_lookup[cell_line][protein] = None
                            elif len(matches) == 0:
                                protein_synonym_lookup[cell_line][protein] = None

            for cell_line, lookup in protein_synonym_lookup.items():
                for protein, synonym in lookup.items():
                    if synonym is not None: 
                        assert protein ==synonym, f"Mismatch in synonym lookup for protein {protein} in cell line {cell_line}"
                
                logger.warning(f"Cell line {cell_line} - validate the following synonyms manually: {validate_synonyms[cell_line]}. NOTE: these synonyms have not been saved. ")
        
        elif mode == "street_et_al":
            raise NotImplementedError("Street et al. mode not yet implemented.")

        self.protein_synonym_lookup = protein_synonym_lookup

    
    def load_rec_y2h_ppi(self): 
        # Rec-Y2H PPI info
        recy2h_results = self.CONFIG["PPI_INFO"]["recy2h_ppi_dir"]["results"]

        # load all rec-Y2H screening results
        recy2h_df = pd.read_excel(recy2h_results)
        # use threshold of sumIS >= 7.1 as in original paper
        recy2h_df = recy2h_df[recy2h_df["sumIS"] >= 7.1]
        recy2h_df["Protein A"] = recy2h_df["Protein A"].str.lower()
        recy2h_df["Protein B"] = recy2h_df["Protein B"].str.lower()

        # Go through each interaction in rec-Y2H and if we have both proteins in our eCLIP data, add to final PPI table
        recy2h_ppi_table = []
        for _, row in recy2h_df.iterrows():
            prot_a = row["Protein A"]
            prot_b = row["Protein B"]

            for cell_line in self.CONFIG["CELL_LINES"]:

                syn_a = self.protein_synonym_lookup[cell_line][prot_a]
                syn_b = self.protein_synonym_lookup[cell_line][prot_b]

                # Only include if both synonyms are found
                if syn_a is not None and syn_b is not None:

                    rbps_sorted = sorted([syn_a, syn_b])
                    interaction = f"{rbps_sorted[0]}-{rbps_sorted[1]}"

                    recy2h_ppi_table.append({
                        "Cell Line": cell_line,
                        "Interaction": interaction,
                        "rec-Y2H | Table S2": True, 
                    })

        recy2h_ppi_table = pd.DataFrame(recy2h_ppi_table)
        return recy2h_ppi_table


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


    def manually_set_num_tasks(self): 
        return 4
    

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
            
            return tuple(sorted([feature + suffix for feature in feature_names]))
    
    
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
            return pl.read_csv(OUTPUT_FILE, separator="\t")

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
                            rbp2, pos2 = None, None

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
            df = df.sort(["Cell Line", "RBP 1", "Position 1", "RBP 2", "Position 2"])

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

            df.write_csv(OUTPUT_FILE, separator="\t")
            logger.success(f"SUCCESS: SHAP average file for '{metric}' saved to '{OUTPUT_FILE}'.")




#########################################################################################################################
# Non-class functions
#########################################################################################################################


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
    
    elif args.metric and args.cell_line and args.start and args.stop:
        
        analyzer = SecondOrderShapNetworkAnalyzer()
        analyzer.calculate_metrics_for_column_range(
            cell_line=args.cell_line,
            metric=args.metric,
            start=args.start,
            stop=args.stop
        )
