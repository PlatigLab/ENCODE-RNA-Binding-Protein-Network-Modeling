""" 
Packages and Literals
"""

import pandas as pd
import glob, os, argparse
import matplotlib.pyplot as plt 
import seaborn as sns 

ayan_shap_folder = "/project/PlatigLab/data/collaborators/BWH/3_XGBDT_SHAP_data_2024_07/"
best_performing_distance_threshold=100

actual_psi_column_name = "target"
predicted_psi_column_name = "psi_hat"

# dictionary to rename the splice junction positions to numbers
splice_junction_position_renaming = {
    "5_left": 1, 
    "5_right": 2, 
    "center_left": 3, 
    "center_right": 4, 
    "3_left": 5, 
    "3_right": 6
}


"""
Argument Parsing
"""
# Create the argument parser
parser = argparse.ArgumentParser(description='Compare double binding SHAP values with single-binders for a specific cell line and RNA binding proteins.')

# Add the arguments
parser.add_argument('--cell_line', type=str, help='The cell line')
parser.add_argument('--ppi_rbp1', type=str, help='The first RNA binding protein')
parser.add_argument('--ppi_rbp2', type=str, help='The second RNA binding protein')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
cell_line = args.cell_line
rbp1 = args.ppi_rbp1.lower()
rbp2 = args.ppi_rbp2.lower()


"""
Get the right files and load the data with just the necessary columns 
"""
# create output file name using arguments
output_file_prefix = f"{cell_line}_{rbp1}_{rbp2}"

files = sorted(
    glob.glob(
        f"{ayan_shap_folder}/**/{cell_line}-{best_performing_distance_threshold}-*data.dat",
        recursive=True
    )
)

files = [file for file in files if "-train-" not in file]
assert len(files)==2, print(files)

# concat the test and validate files (just for column names) and get the columns as a list 
columns = pd.concat([ pd.read_csv(file, sep=",", nrows=0) for file in files ]).columns.to_list()
# get only columns related to rbps that are actually interacting 
# finally, add the actual and predicted PSI values to the column
use_cols = [column for column in columns if column.split("_")[0].lower() in [rbp1, rbp2]] + [actual_psi_column_name, predicted_psi_column_name]
assert len(use_cols)==((6*2*2)+2), print(f"{args} {use_cols}")

data_df = pd.concat([ pd.read_csv(file, sep=",", usecols=use_cols) for file in files ])


"""
Create a dictionary to store the data for plotting
"""
plotting_dict = {}

for position in splice_junction_position_renaming: 
    
    # for each column, check if the position is in the column name and if so, check the rbp is one of the interacting RBPs
    # finally, add the actual and predicted PSI values to the column
    subset_cols = [column for column in use_cols if position in "_".join(column.split("_")[1:])] + [actual_psi_column_name, predicted_psi_column_name]
    assert len(subset_cols)==6, print(f"{args} {subset_cols}")

    binding_cols = sorted([column for column in subset_cols if column.endswith(position)])
    shap_cols = sorted([column for column in subset_cols if column.endswith(f"{position}_shap")])
    assert len(binding_cols)==2 and len(shap_cols)==2

    # subset data_df with subset_cols
    subset_df = data_df[subset_cols].copy(deep=True)

    double_binding = subset_df[(subset_df[binding_cols[0]]==1) & (subset_df[binding_cols[1]]==1)]
    single_binding = subset_df[(subset_df[binding_cols[0]]==1) ^ (subset_df[binding_cols[1]]==1)]

    if len(double_binding)==0: 
        plotting_dict[splice_junction_position_renaming[position]] = None

    else:
    
        plotting_df = []

        for row in pd.concat([double_binding, single_binding]).itertuples(): 

            binding_mode = [ binding_col.split("_")[0] for binding_col in binding_cols if getattr(row, binding_col)==1 ]
            assert len(binding_mode)>=0 and len(binding_mode)<=2, print(binding_mode)
            
            if len(binding_mode) == 2: 
                binding_mode = "Both"            
            elif len(binding_mode) == 1: 
                binding_mode = binding_mode[0]
            
            for shap_col in shap_cols:
                tmp_rbp = shap_col.split("_")[0]

                plotting_df.append(
                    [
                        binding_mode, 
                        tmp_rbp, 
                        getattr(row, shap_col)

                    ]
                )

        plotting_df = pd.DataFrame(plotting_df, columns=["Binding Mode", "RBP", "SHAP"])

        categorical_order = ["Both"] + sorted(plotting_df["RBP"].unique().tolist())
        plotting_df["Binding Mode"] = pd.Categorical(plotting_df["Binding Mode"], categories=categorical_order, ordered=True)
        plotting_df = plotting_df.sort_values(["Binding Mode", "RBP"])

        plotting_dict[splice_junction_position_renaming[position]] = plotting_df.copy(deep=True)


"""
Plotting
"""
if all([plotting_dict[position] is None for position in plotting_dict]) == False:

    nrows = 2
    ncols = 3

    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=(20, 10), dpi=400,) #sharex=True, sharey=True)

    counter = 1
    for row_num in range(nrows):
        for col_num in range(ncols):
            ax = axes[row_num, col_num]
            df = plotting_dict[counter]

            ax.set_title(f"Position {counter}", fontsize=20, pad=5)

            if df is not None:
                sns.boxplot(data=df, x="Binding Mode", y="SHAP", hue="RBP", ax=ax)
                ax.tick_params(axis='x', labelsize=16)
                ax.tick_params(axis='y', labelsize=16)

            counter += 1

    fig.suptitle(f"{cell_line}: {rbp1.upper()}-{rbp2.upper()} \nDouble Binding vs Single Binding SHAP Values", fontsize=30, y=1.02)
    fig.supxlabel("Binding Mode", fontsize=24)
    fig.supylabel("SHAP Value", fontsize=24)


    plt.tight_layout()
    plt.savefig(f"../outputs/plots/{output_file_prefix}_test.png", dpi=400, bbox_inches="tight")
    plt.close()