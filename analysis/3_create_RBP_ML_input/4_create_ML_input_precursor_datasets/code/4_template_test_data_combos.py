import yaml, pandas

# Path to the YAML file
yaml_file = "/project/PlatigLab/users/yogi/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/6_create_RBP_to_PSI_input_ML_datasets/final_modeling_input_datasets/data_config.yaml"


# read count quantiles, conditions, and PSI subsets
read_count_quantiles = [0, .1, .25, .5]
conditions = [None, "CTRL"]
# NOTE: the -0.1 is to just a placeholder and meant 
# only to remove PSI values greater than 1
PSI_subsets = [
    [-0.1, 1], 
    [0, 1], 
    [0.1, 0.9]
]


# Read the YAML file
with open(yaml_file, 'r') as file:
    config = yaml.safe_load(file)

# for cell line 
for cell_line in config['Data Files']:

    # for distance peak threshold 
    for distance_threshold in config['Data Files'][cell_line]: 

        # for data creation method
        for data_creation_method in config['Data Files'][cell_line][distance_threshold]:

            # get the single file belonging to this combination 
            input_ML_file = config['Data Files'][cell_line][distance_threshold][data_creation_method]
            assert len(input_ML_file) == 1, "{} {} {}".format(cell_line, distance_threshold, data_creation_method)

            # Read the input file
            # THIS WILL PROBABLY TAKE A LONG TIME BECAUSE THE FILE IS MASSIVE
            ML_data = pandas.read_csv(input_ML_file, sep="\t", compression='gzip', index_col=0)

            # for each condition
            for condition in conditions:
                
                # just keep all the data if no condition is specified
                if condition is None:
                    condition_df = ML_data.copy(deep=True)

                # for the CTRL-only subsetting
                elif condition is not None:
                    # subset the data to the condition
                    condition_df = ML_data[ML_data["RBP_KD_Target"] == condition].copy(deep=True)

                # for each read count quantile value 
                for read_count_quantile in read_count_quantiles:

                    # Subset the DataFrame for values above the read count quantile
                    read_count_df = condition_df[condition_df['Total Read Counts'] > condition_df['Total Read Counts'].quantile(read_count_quantile)]

                    # for each PSI subset
                    for PSI_subset_group in PSI_subsets:
                        
                        # take the PSI values at the specified cutoffs
                        input_df = read_count_df[
                            (read_count_df["Target_PSI"] >= PSI_subset_group[0]) & 
                            (read_count_df["Target_PSI"] <= PSI_subset_group[1])
                        ]

                        # run all models and hyperparameters with given data 
                        run_all_models_and_hyperparameters(
                            input_df, 
                            target_variable="Target_PSI"
                        )




                

