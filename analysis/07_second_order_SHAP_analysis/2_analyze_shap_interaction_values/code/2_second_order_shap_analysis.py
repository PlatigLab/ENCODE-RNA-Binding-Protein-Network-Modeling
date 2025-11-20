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



        

