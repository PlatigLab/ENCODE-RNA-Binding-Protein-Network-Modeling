import polars as pl, glob, matplotlib.pyplot as plt, pandas as pd, re, pathlib, json, matplotlib.colors as mcolors, concurrent.futures, tqdm, os, seaborn as sns

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
    LINEAR_MODEL_DIR = "/project/PlatigLab/data/collaborators/BWH/5_linear_and_xgbdt_models_2024_10/linear-models-2024-10/linear-models-100-3a00c07e/"
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
        self.load_RBP_PPI_pairs()
        self.retrieve_rbp_ppi_events_and_controls()


    def load_RBP_PPI_pairs(self):

        with open(self.rbp_comparisons_file, "r") as f:
            self.rbp_ppi = json.load(f)
            
        logger.success(f"FROM CACHE: RBP PPI info loaded")


    def load_SHAP_data(self):
        logger.info(f"FROM CACHE: Loading SHAP data for distance threshold: {self.distance_threshold}.")
        
        shap_data = {}

        for cell_line in self.cell_lines: 

            shap_file = f"{self.FEATHER_CACHE_DIR}/{cell_line}-{self.distance_threshold}-shap_data.feather"
            shap_data[cell_line] = pl.read_ipc(shap_file)
        
        self.shap_data = shap_data

        logger.success(f"Loaded SHAP data for distance threshold: {self.distance_threshold}")
    

    def load_linear_model_results(self): 
        logger.info(f"FROM CACHE: Loading linear model predictions for distance threshold: {self.distance_threshold}.")

        linear_coefficients = {}
        for cell_line in self.cell_lines: 

            linear_coefficients_file = glob.glob(f"{self.LINEAR_MODEL_DIR}/{cell_line}-{self.distance_threshold}-*-linear-model-beta.dat")
            assert len(linear_coefficients_file) == 1

            linear_coefficients[cell_line] = pd.read_csv(linear_coefficients_file[0], sep=",", index_col=0)
            linear_coefficients[cell_line].index.name = "Feature"
        
        self.linear_coefficients = linear_coefficients

        linear_model_results = {}
        for cell_line in self.cell_lines:
            linear_model_results_file = f"{self.FEATHER_CACHE_DIR}/{cell_line}_{self.distance_threshold}_linear_model_results.feather"
            linear_model_results[cell_line] = pl.read_ipc(linear_model_results_file)

        self.linear_model_results = linear_model_results

        logger.success(f"Retrieved linear model predictions for distance threshold:  {self.distance_threshold}.")


    def get_number_SLURM_CPUs(self):
        slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")

        if slurm_cpus is not None:
            return int(slurm_cpus)
        else:
            logger.warning("SLURM_CPUS_PER_TASK environment variable not set. Defaulting to 1 CPU.")
            return 1


    def _cache_to_featherv2(self, df, file_path):

        if not pathlib.Path(file_path).exists():
            logger.info(f"Caching to Feather V2 (Arrow): {file_path}")

            df.write_ipc(file_path, compression="lz4")
            logger.success(f"Finished creating Feather V2 (Arrow) file for {file_path}")

        else:
            logger.warning(f"Feather V2 (Arrow) file already exists: {file_path}")


    def retrieve_rbp_cobinding(self, data, rbp_ppi_pair, position1, position2):
        rbp1, rbp2 = rbp_ppi_pair

        search_columns = [f"{rbp1.upper()}_{position1}", f"{rbp2.upper()}_{position2}"]

        assert all(col in data.columns for col in search_columns), logger.error(f"Columns {search_columns} not found in DataFrame")

        tmp_data = data.filter(
            (pl.col(search_columns[0]) == 1) & (pl.col(search_columns[1]) == 1)
        ).select("index")

        tmp_data = tmp_data.with_columns(
            [
                pl.lit(rbp1).alias("RBP 1"),
                pl.lit(rbp2).alias("RBP 2"),
                pl.lit(self.splice_junction_position_renaming[position1]).alias("Position 1"),
                pl.lit(self.splice_junction_position_renaming[position2]).alias("Position 2")
            ]
        )

        return tmp_data


    def retrieve_rbp_ppi_events_and_controls(self): 

        if len(glob.glob(f"{self.PPI_CACHE_DIR}/{self.cell_line}-{self.distance_threshold}-*")) == 2:
            pass
        
        else: 

            logger.info("Retrieving RBP PPI events and controls.")

            if not hasattr(self, 'linear_model_results'):
                self.load_linear_model_results()

            if not hasattr(self, 'shap_data'):
                self.load_SHAP_data()

            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.get_number_SLURM_CPUs()) as executor:
                true_ppi = {
                    executor.submit(self.retrieve_rbp_cobinding, self.linear_model_results, rbp_ppi_pair, position_key, position_key): (rbp_ppi_pair, position_key) for rbp_ppi_pair in self.rbp_ppi for position_key in self.splice_junction_position_renaming.keys()
                }

                for future in tqdm.tqdm(concurrent.futures.as_completed(true_ppi), total=len(true_ppi), desc="True PPI Events"):
                    results.append(future.result())
            
            true_ppi = pl.concat(results, how="vertical").unique()

            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.get_number_SLURM_CPUs()) as executor:
                futures = []
                for rbp1, rbp2 in self.rbp_ppi:

                    for position1 in self.splice_junction_position_renaming.keys():
                        for position2 in self.splice_junction_position_renaming.keys():
                            if position1 != position2:

                                futures.append(executor.submit(self.retrieve_rbp_cobinding, self.linear_model_results, (rbp1, rbp2), position1, position2))

                for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Mismatched PPI Events"):
                    results.append(future.result())

            mismatched_ppi = pl.concat(results, how="vertical").unique()

            true_ppi_indices = set(true_ppi["index"].to_list())

            mismatched_ppi_indices = set(mismatched_ppi["index"].to_list())
            mismatched_ppi_indices = mismatched_ppi_indices.difference(true_ppi_indices)

            for model_type, df in [("linear", self.linear_model_results), ("xgboost", self.shap_data)]:

                df = df.join(true_ppi, on="index", how="left")
                df = df.join(mismatched_ppi, on="index", how="left")

                df = df.with_columns(
                    pl.when(pl.col("index").is_in(true_ppi_indices))
                    .then(pl.lit("True PPI"))
                    .when(pl.col("index").is_in(mismatched_ppi_indices))
                    .then(pl.lit("Mismatched PPI"))
                    .otherwise(pl.lit("Non-PPI"))
                    .alias("PPI Type")
                )

                assert df["PPI Type"].has_nulls() == False

                self._cache_to_featherv2(df, f"{self.PPI_CACHE_DIR}/{self.cell_line}-{self.distance_threshold}-{model_type}-ppi_events_and_controls.feather")
        
        logger.success("Finished retrieving RBP PPI events and controls.")


    def compare_xgboost_and_linear_model_predictions(self): 
        logger.info("Plotting XGBoost vs Linear Model prediction performance.")

        if not hasattr(self, 'shap_data'):
            self.load_SHAP_data()

        if not hasattr(self, 'linear_model_results'):
            self.load_linear_model_results()

        test_xgboost_predictions = self.shap_data.filter(pl.col("Data Partition") == "test")
        test_lm_predictions = self.linear_model_results.filter(pl.col("Data Partition") == "test")

        assert test_lm_predictions.shape[0] == test_xgboost_predictions.shape[0]
        logger.info(f"{test_lm_predictions.shape[0]} examples used to evaluate each model")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=200)

        for ax, (title, predictions) in zip(axes, [("Linear Regression", test_lm_predictions), ("XGBoost", test_xgboost_predictions)]):
            ax.scatter(predictions["target"], predictions["psi_hat"], facecolors='none', edgecolors='blue', alpha=0.01, s=0.1)

            ax.set_title(title)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")

            # Calculate R2 score
            r2 = r2_score(predictions["target"], predictions["psi_hat"])

            # Add y=x line
            ax.plot([0, 1], [0, 1], color='green', linestyle='--', linewidth=2)

            # Add number of points and R2 score to the plot
            num_points = len(predictions)
            ax.text(0.05, 0.95, f"# Points: {num_points}\nR2: {r2:.2f}", transform=ax.transAxes, verticalalignment='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        plt.suptitle(f"{self.cell_line} {self.distance_threshold}: Test Set Prediction Performance (Linear vs XGBoost Models)", fontsize=16)
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=200, sharex=True,)

        for ax, (title, predictions) in zip(axes, [("Linear Regression", test_lm_predictions), ("XGBoost", test_xgboost_predictions)]):
            hb = ax.hist2d(predictions["target"], predictions["psi_hat"], bins=100, cmap='Blues', norm=mcolors.LogNorm())
            cbar = plt.colorbar(hb[3], ax=ax)
            cbar.set_label('Logarithm Density')

            ax.set_title(title)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")

            # Calculate R2 score
            r2 = r2_score(predictions["target"], predictions["psi_hat"])

            # Add y=x line
            ax.plot([0, 1], [0, 1], color='green', linestyle='--', linewidth=2)

            # Add number of points and R2 score to the plot
            num_points = len(predictions)
            ax.text(0.05, 0.95, f"# Points: {num_points}\nR2: {r2:.2f}", transform=ax.transAxes, verticalalignment='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        plt.suptitle(f"{self.cell_line} {self.distance_threshold}: Test Set Prediction Performance (Linear vs XGBoost Models)\nNOTE: logarithmic density used for coloring", fontsize=16, y=1.01)

        plt.tight_layout()
        plt.show()
        

    #TODO get Ayan to give predictions for testing set for linear models
    def compare_rbp_ppi_performance_xgboost_vs_linear_model(self): 
        logger.info("Comparing RBP PPI performance between XGBoost and Linear Models.")
        
        if not hasattr(self, 'shap_data'):
            self.load_SHAP_data()

        if not hasattr(self, 'linear_model_results'):
            self.load_linear_model_results()

        test_xgboost_predictions = self.shap_data.filter(pl.col("Data Partition") == "test")
        test_lm_predictions = self.linear_model_results.filter(pl.col("Data Partition") == "test")

        r2_test_lm = r2_score(test_lm_predictions["target"], test_lm_predictions["psi_hat"])
        r2_test_xgboost = r2_score(test_xgboost_predictions["target"], test_xgboost_predictions["psi_hat"])

        assert test_lm_predictions.shape[0] == test_xgboost_predictions.shape[0]
        logger.info(f"{test_lm_predictions.shape[0]} examples used to evaluate each model")

        linear_model_ppi_predictions = self.retrieve_rbp_ppi_events_and_controls(df=test_lm_predictions)
        xgboost_model_ppi_predictions = self.retrieve_rbp_ppi_events_and_controls(df=test_xgboost_predictions)

        for key in linear_model_ppi_predictions.keys():
            assert linear_model_ppi_predictions[key].shape[0] == xgboost_model_ppi_predictions[key].shape[0]

        r2_scores = {
            "Model": [],
            "PPI Category": [],
            "R2 Score": []
        }
        
        for key in linear_model_ppi_predictions.keys():

            for plot_type in ["hist", "scatter"]:

                fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=200,)

                for ax, (title, predictions) in zip(axes, [("Linear Regression", linear_model_ppi_predictions[key]), ("XGBoost", xgboost_model_ppi_predictions[key])]):
                    
                    if plot_type == "hist":
                        hb = ax.hist2d(predictions["target"], predictions["psi_hat"], bins=100, cmap='Blues', norm=mcolors.LogNorm())
                        cbar = plt.colorbar(hb[3], ax=ax)
                        cbar.set_label('Logarithm Density')
                    elif plot_type == "scatter":
                        ax.scatter(predictions["target"], predictions["psi_hat"], facecolors='none', edgecolors='blue', alpha=0.1, s=0.5)

                    ax.set_title(title)
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")

                    ax.plot([0, 1], [0, 1], color='green', linestyle='--', linewidth=2)

                    # Calculate R2 score
                    r2 = r2_score(predictions["target"], predictions["psi_hat"])
                    
                    r2_scores["Model"].append(title)
                    r2_scores["PPI Category"].append(key)
                    r2_scores["R2 Score"].append(r2)

                    # Add number of points and R2 score to the plot
                    num_points = len(predictions)
                    ax.text(0.05, 0.95, f"# Points: {num_points}\nR2: {r2:.2f}", transform=ax.transAxes, verticalalignment='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

                plt.suptitle(f"{self.cell_line} {self.distance_threshold} Test Set: {key}", fontsize=20)
                plt.tight_layout()
                plt.show()

        r2_scores = pd.DataFrame.from_dict(r2_scores, orient="columns")

        plt.figure(dpi=200, figsize=(8,3))

        sns.barplot(data=r2_scores, y="PPI Category", x="R2 Score", hue="Model", palette=["tomato", "royalblue"], orient="h", width=0.6)

        plt.axvline(x=r2_test_lm, color='tomato', linestyle='-', linewidth=2, label=f'Linear Model R2: {r2_test_lm:.2f}')
        plt.axvline(x=r2_test_xgboost, color='royalblue', linestyle='-.', linewidth=2, label=f'XGBoost R2: {r2_test_xgboost:.2f}')

        plt.title(f"{self.cell_line} {self.distance_threshold}: Test Set R2 Scores\nby PPI Category/Model", fontsize=16, pad=20)
        plt.ylabel("PPI Category", fontsize=12, labelpad=10)
        plt.xlabel("R2 Score", fontsize=12, labelpad=10)
        plt.legend(title="Model/R2 Scores", fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.show()
