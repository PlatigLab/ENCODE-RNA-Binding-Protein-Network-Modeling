##################
# Load libraries #
##################
import pandas as pd
import sys 

############
# Literals #
############
columns_to_drop = ["psip", "delta_psi", "FDR", "p_value", "RBP_KD", "has_RBP_KD"]

#############
# Main Code #
#############

# get first argument from command line
input_file = sys.argv[1]

# read in data by specifying the compression type and separator
input = pd.read_csv(
    input_file, 
    compression="gzip",
    sep=",", 
)

# subset data to only include control samples
input = input[input["RBP_KD"]=="NONE"]

# remove columns that are not needed
input = input.drop(columns=columns_to_drop)

# replace -1 values with 0 
input = input.replace(-1, 0)

# drop duplicates 
input = input.drop_duplicates()

# output file with gunzip compression 
input.to_csv(
    "../output/control_only_cleaned_input_binding_data/{}.csv.gz".format(
        input_file.split("/")[-2]
    ),
    compression="gzip",
    index=False, 
)