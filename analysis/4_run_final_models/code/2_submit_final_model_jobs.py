import json, os, copy

SCRIPT_FILE="./1_xgbregressor_on_all_data.py"
MODEL_CONFIGS = "../../3_choose_dataset_and_model_parameters/output/chosen_model_hyperparameters/final_model_hyperparameters.json"

CPUS = 32
MEMORY= 256
PARTITION="standard"
ACCOUNT="platiglab"
PYTHON_EXEC="/bin/python3.11"
SLURM_LOGS_DIR = "../outputs/SLURM_logs"

if __name__ == "__main__":
    # Read the model configs
    with open(MODEL_CONFIGS, "r") as f:
        model_configs = json.load(f)
    
    # Loop over the models and submit jobs
    for i, model_config in enumerate(model_configs):
        for random_seed in range(100, 1000, 100): 

            input_config = copy.deepcopy(model_config)
            input_config["random_seed"] = random_seed
            
            # Filter keys that don't contain "_r2_"
            filtered_args = {k: v for k, v in input_config.items() if "_r2_" not in k}
            # Sort the dictionary by key
            sorted_args = dict(sorted(filtered_args.items()))
            # Concatenate key-value pairs into a single string separated by '-'
            concatenated_args = '-'.join(f"{key}-{value}" for key, value in sorted_args.items())

            input_config['unique_id'] = concatenated_args

            # Write the input_config to a JSON file
            config_filename = f"{concatenated_args}.json"
            with open(config_filename, "w") as config_file:
                json.dump(input_config, config_file, indent=4)

            # Update the command to pass the JSON file
            cmd = f"sbatch --job-name={concatenated_args} --output={SLURM_LOGS_DIR}/{concatenated_args}.out --error={SLURM_LOGS_DIR}/{concatenated_args}.err --cpus-per-task={CPUS} --mem={MEMORY}GB --partition={PARTITION} --account={ACCOUNT} --wrap='{PYTHON_EXEC} {SCRIPT_FILE} --config_file {config_filename}'"

            print(f"Submitting job: {cmd}\n")
            os.system(cmd)
