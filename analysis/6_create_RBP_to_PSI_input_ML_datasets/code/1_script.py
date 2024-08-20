# %% [markdown]
# # TODO

# %% [markdown]
# ## Purpose: 
# 
# {TODO}

# %%
import pandas as pd
# pd.set_option('display.max_rows', 1000000)
# pd.set_option('display.max_columns', 1000000)
# pd.set_option('display.max_colwidth', 10000000)

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

import glob, gzip, pickle, sys

# %% [markdown]
# ## Literals
# 

cell_line = str(sys.argv[1])
threshold = int(sys.argv[2])

# %%
rmats_data_path = "/project/PlatigLab/data/collaborators/BWH/1_ENCODE_shRNA_RBP_KD_2024-04-hg38-gencode-v29/"

rmats_file_column_subset = ["chr", "strand", "exonStart_0base", "exonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE", "IJC_SAMPLE_1", "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2", "IncLevel1", "IncLevel2"]

exon_ordering = {
    "+": {
        "1": "upstreamES", 
        "2": "upstreamEE",
        "3": "exonStart_0base", 
        "4": "exonEnd", 
        "5": "downstreamES", 
        "6": "downstreamEE"
    }, 
    "-": {
        "1": "downstreamEE", 
        "2": "downstreamES",
        "3": "exonEnd",
        "4": "exonStart_0base", 
        "5": "upstreamEE", 
        "6": "upstreamES"
    }
}

sample_names = ["KD-1", "KD-2", "CTRL-1", "CTRL-2"]

# %% [markdown]
# ## Defining Important Variables
# 
# * RBPs
# * KD-to-Control Accession IDs
# * Expression Values

# %% [markdown]
# #### Get the RBPs per Cell Line that we are going to look at

# %%
# key is cell line and value is list of rbps
selected_rbps = []

file = glob.glob("../../5_assign_eCLIP_to_splice_junctions/output/bedtools_input/*{}*_sorted.bed".format(cell_line))
assert len(file)==1

tmp_df = pd.read_csv(file[0], sep="\t", header=None) 

selected_rbps = sorted(tmp_df[3].str.split("_").str[0].unique().tolist())

# %% [markdown]
# #### Get the Control Accession values per RBP KD

# %%
control_associations = {}

file = glob.glob("../../3_get_expression_shrna_BAMs/output/2_final_raw_counts_matrices/{}*associations*".format(cell_line))
assert len(file)==1

tmp_df = pd.read_csv(file[0], sep="\t")
    
control_associations = tmp_df.set_index("RBP KD").to_dict()["Control Accession"]

# %% [markdown]
# #### Get gene expression

# %%


# %% [markdown]
# ## Load and Subset `rMATS Skipped Exon` Files

# %%

# for every SE file 
file = str(sys.argv[3])

# get rbp from the file path
rbp = file.split("/")[-2].split("-")[0]
# paranoia check: the cell line should be the same as what's labelled on the folder 
assert file.split("/")[-2].split("-")[2] == cell_line, print(file.split("/")[-2].split("-")[2], cell_line)

# select only for RBPs we are looking at and for polyA mRNA samples
if "-Transfection-" not in file and rbp in selected_rbps: 

    # read rMATS file and subset for relevant columns
    input_df = pd.read_csv(file, sep="\t")[rmats_file_column_subset]
    # add RBP KD column 
    input_df["RBP_KD_Target"] = rbp
    


# %% [markdown]
# ## Separate `PSI` and `Total Counts` Into Separate Columns

# %%
    
input_df["PSI_KD-1"] = input_df["IncLevel1"].str.split(",").str[0]
input_df["PSI_KD-2"] = input_df["IncLevel1"].str.split(",").str[1]

input_df["PSI_CTRL-1"] = input_df["IncLevel2"].str.split(",").str[0]
input_df["PSI_CTRL-2"] = input_df["IncLevel2"].str.split(",").str[1]

input_df = input_df.drop(columns = ["IncLevel1", "IncLevel2"])

input_df["Counts_KD-1"] = (input_df["IJC_SAMPLE_1"].str.split(",").str[0]).astype("int16") + (input_df["SJC_SAMPLE_1"].str.split(",").str[0]).astype("int16")
input_df["Counts_KD-2"] = (input_df["IJC_SAMPLE_1"].str.split(",").str[1]).astype("int16") + (input_df["SJC_SAMPLE_1"].str.split(",").str[1]).astype("int16")
input_df["Counts_CTRL-1"] = (input_df["IJC_SAMPLE_2"].str.split(",").str[0]).astype("int16") + (input_df["SJC_SAMPLE_2"].str.split(",").str[0]).astype("int16")
input_df["Counts_CTRL-2"] = (input_df["IJC_SAMPLE_2"].str.split(",").str[1]).astype("int16") + (input_df["SJC_SAMPLE_2"].str.split(",").str[1]).astype("int16")

input_df = input_df.to_dict(orient="records")

# %% [markdown]
# ## Load `Splice Junction to # RBP Peaks` Data

# %%
junction_to_num_peaks = {}

with gzip.GzipFile("../../5_assign_eCLIP_to_splice_junctions/output/splice_junction_rbp_num_peaks/all_RBP_peaks_num_per_splice_junction.pkl.gz", 'rb') as in_file: 
    junction_to_num_peaks = pickle.load(in_file)
    junction_to_num_peaks = junction_to_num_peaks[cell_line]


# %% [markdown]
# ## Create Input Data for ML Model by Starting w/ Creating # Peaks per RBP per Sample

# %% [markdown]
# #### Pseudocode of Algorithm

# %%
# for cell line: 
#     for distance threshold: 
                
#         for each row in concat rmats dict: 
#             get the correct orientation of which column corresponds to 1-6 in our paradigm 
            
#             for each sample in row: 
#                 if not "nan": 
                    
#                     create unique_id: should be chr, strand, all 6 splice junction coordinates for that event, sample name, and RBP KD (if applicable)
#                     (for controls, use control accession lookup table to convert to ENCSR ID and do not include RBP KD)
                    
#                     check that unique_id has not been encountered prior 
                    
#                     if new unique_id: 
                        
#                         save unique_id to encountered ids 

#                         for each position (1-6): 
                            
#                             create lookup string: chr_splice-jnction-coordinate_strand
                            
#                             for each rbp that we are looking at in that cell line: 
#                                 use lookup string to save number of peaks total 
                
#                                 saving should be done using dictionary structure such as: 
#                                     {
#                                         unique_id: {
#                                             total counts: value
#                                             RBP_1_binding: value 
#                                         }
#                                     }
                        
#                         save "chr", inclevel (target), total counts,
                            
#         make sure that all columns that are supposed to be numeric are casted as such 

# %%

# dict corresponds to creating a single cell-line-and-threshold specific dataset 
# where the key is the unique_id described above in the pseudocode and value is 
# each feature per skipped exon per sample
ML_input_data = {}

events_encountered = set()
    
for row in input_df:
    position_definition = exon_ordering[row["strand"]]
    
    for sample in sample_names: 
        
        if row["PSI_"+sample] != "NA": 
            
            unique_id = ""
            
            unique_id = "_".join(
                [str(row[coord_column]) for coord_column in rmats_file_column_subset[0:7]]
            )
            
            if "KD" in sample: 
                kd_ctrl_string = row["RBP_KD_Target"] 
            elif "CTRL" in sample: 
                kd_ctrl_string = control_associations[row["RBP_KD_Target"]]
            
            unique_id = "_".join(
                [unique_id, kd_ctrl_string, sample]
            )
                                
            if unique_id not in events_encountered: 
                
                events_encountered.add(unique_id)
                
                ML_input_data[unique_id] = {}
                
                for position in position_definition: 
                    
                    splice_junction_id = "_".join(
                        [row["chr"], str(row[position_definition[position]]), row["strand"]]
                    )
                    
                    if splice_junction_id in junction_to_num_peaks: 
                        junction_present=True
                    else: 
                        junction_present=False
                        
                    for rbp in selected_rbps: 

                        feature_string = "_".join(
                            [rbp, position, "binding"]
                        ) 

                        if junction_present: 
                            ML_input_data[unique_id][feature_string] = junction_to_num_peaks[splice_junction_id][rbp]
                            
                        elif not junction_present: 
                            ML_input_data[unique_id][feature_string] = 0
                
                ML_input_data[unique_id]["chr"] = row["chr"]
                                        
                if "KD" in sample: 
                    ML_input_data[unique_id]["RBP_KD_Target"] = row["RBP_KD_Target"]
                elif "CTRL" in sample: 
                    ML_input_data[unique_id]["RBP_KD_Target"] = "CTRL"                        
                
                if sample=="KD-1": 
                    ML_input_data[unique_id]["Inclusion Counts"] = int(row["IJC_SAMPLE_1"].split(",")[0])
                    ML_input_data[unique_id]["Skipping Counts"] = int(row["SJC_SAMPLE_1"].split(",")[0])
                
                elif sample=="KD-2": 
                    ML_input_data[unique_id]["Inclusion Counts"] = int(row["IJC_SAMPLE_1"].split(",")[1])
                    ML_input_data[unique_id]["Skipping Counts"] = int(row["SJC_SAMPLE_1"].split(",")[1])
                    
                elif sample=="CTRL-1": 
                    ML_input_data[unique_id]["Inclusion Counts"] = int(row["IJC_SAMPLE_2"].split(",")[0])
                    ML_input_data[unique_id]["Skipping Counts"] = int(row["SJC_SAMPLE_2"].split(",")[0])
                    
                elif sample=="CTRL-2": 
                    ML_input_data[unique_id]["Inclusion Counts"] = int(row["IJC_SAMPLE_2"].split(",")[1])
                    ML_input_data[unique_id]["Skipping Counts"] = int(row["SJC_SAMPLE_2"].split(",")[1])
                
                ML_input_data[unique_id]["Total Read Counts"] = row["Counts_" + sample]

                ML_input_data[unique_id]["Target_PSI"] = row["PSI_"+sample]


ML_input_data = pd.DataFrame.from_dict(ML_input_data, orient="index")

output_file = "/project/PlatigLab/users/yogi/ENCODE-RNA-Binding-Protein-Network-Modeling/analysis/6_create_RBP_to_PSI_input_ML_datasets/output/{}_{}_{}_initial-dataset.tsv.gz".format(file.split("/")[-2].split("-")[0], cell_line, threshold)

ML_input_data.to_csv(output_file, sep="\t", compression="gzip")

print("completed")

