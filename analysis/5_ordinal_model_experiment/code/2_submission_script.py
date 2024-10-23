import yaml, copy, subprocess

WANDB_PROJECT_NAME = "yogi-OLS-baseline-regression"
PARAMS_FILE = "./params.yaml"
MODEL_TYPE = "OLSRegression"

JOB_CONFIG_DIR = "../output/configs"
SLURM_LOGS_DIR = "../output/SLURM_logs"

with open(PARAMS_FILE, 'r') as file:
    params = yaml.safe_load(file)


def make_base_config(): return copy.deepcopy(params)


COUNTER = 1
def save_config_and_submit_SLURM_job(config):
    global COUNTER

    config["modeling"]["model"] = MODEL_TYPE
    config_yaml_file = f"{JOB_CONFIG_DIR}/{COUNTER}.yaml"

    with open(config_yaml_file, 'w') as file:
        yaml.dump(config, file)

    if config["dataset"]["data_flavor"] == "binary" or config["dataset"]["data_flavor"] == "num_peaks":
        slurm_memory = "100GB"    
    elif config["dataset"]["data_flavor"] == "rbp_exp" or config["dataset"]["data_flavor"] == "rbp_exp_peak":
        slurm_memory = "150GB"

    slurm_command = f"sbatch --mem={slurm_memory} --account=platiglab --partition=standard --time=1-00:00:00 --output={SLURM_LOGS_DIR}/{COUNTER}.out --error={SLURM_LOGS_DIR}/{COUNTER}.err --wrap='python ./1_yogi_model_runner.py --yaml-file-path {config_yaml_file} --wandb-project-name {WANDB_PROJECT_NAME}'"
    subprocess.run(slurm_command, shell=True, check=True)

    COUNTER += 1


for cell_line in params["dataset"]["cell_line"]: 
    for distance in params["dataset"]["distance"]: 
        for control_only in params["dataset"]["control_only"]: 
            for psi_subset in params["dataset"]["psi_subset"]: 
                for read_count_quantile in params["dataset"]["read_count_quantile"]: 
                    for random_state in params["modeling"]["random_state"]:
                        for data_flavor in params["dataset"]["data_flavor"]: 

                            if data_flavor == "binary" or data_flavor == "num_peaks":

                                base_config = make_base_config()

                                base_config["dataset"]["cell_line"] = cell_line
                                base_config["dataset"]["distance"] = distance
                                base_config["dataset"]["control_only"] = control_only
                                base_config["dataset"]["psi_subset"] = psi_subset
                                base_config["dataset"]["read_count_quantile"] = read_count_quantile
                                base_config["modeling"]["random_state"] = random_state
                                base_config["dataset"]["data_flavor"] = data_flavor
                                base_config["dataset"]["count_normalization_method"] = "tmm"
                                base_config["dataset"]["log_counts"] = False

                                save_config_and_submit_SLURM_job(base_config)
                            
                            elif data_flavor == "rbp_exp" or data_flavor =="rbp_exp_peak": 

                                for count_normalization_method in params["dataset"]["count_normalization_method"]: 
                                    for log_counts in params["dataset"]["log_counts"]: 

                                        base_config = make_base_config()

                                        base_config["dataset"]["cell_line"] = cell_line
                                        base_config["dataset"]["distance"] = distance
                                        base_config["dataset"]["control_only"] = control_only
                                        base_config["dataset"]["psi_subset"] = psi_subset
                                        base_config["dataset"]["read_count_quantile"] = read_count_quantile
                                        base_config["modeling"]["random_state"] = random_state
                                        base_config["dataset"]["data_flavor"] = data_flavor
                                        base_config["dataset"]["count_normalization_method"] = count_normalization_method
                                        base_config["dataset"]["log_counts"] = log_counts

                                        save_config_and_submit_SLURM_job(base_config)