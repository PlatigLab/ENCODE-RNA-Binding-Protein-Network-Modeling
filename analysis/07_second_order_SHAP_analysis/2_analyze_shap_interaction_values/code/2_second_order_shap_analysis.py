import yaml, pathlib, glob, json

import pandas as pd, polars as pl

from dataclasses import dataclass
from loguru import logger
from matplotlib_venn import venn2 as mpl_venn2



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

