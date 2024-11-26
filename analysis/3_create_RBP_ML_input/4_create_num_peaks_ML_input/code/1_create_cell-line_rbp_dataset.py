############################################
# Packages #
############################################

import pandas as pd
import glob, gzip, pickle, sys, argparse, copy, numpy

############################################
# Argument parsing #
############################################
parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('--cell_line', type=str, help='Cell line')
parser.add_argument('--rbp', type=str, help='RNA Binding Protein')
parser.add_argument('--threshold', type=int, help='Threshold value')

args = parser.parse_args()

cell_line = args.cell_line
rbp = args.rbp
threshold = args.threshold


############################################
# Literals #
############################################
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


############################################
# Get the RBPs chosen for the cell line  #
############################################
file = glob.glob("../../3_assign_eCLIP_to_splice_junctions/output/bedtools_input/*{}*_sorted.bed".format(cell_line))
assert len(file)==1

tmp_df = pd.read_csv(file[0], sep="\t", header=None) 

selected_rbps = sorted(tmp_df[3].str.split("_").str[0].unique().tolist())


############################################
# Get the association of RBP KD to control experiments #
############################################
control_associations = {}

file = glob.glob("../../1_get_expression_from_BAMs/output/2_final_raw_counts_matrices/{}*associations*".format(cell_line))
assert len(file)==1

tmp_df = pd.read_csv(file[0], sep="\t")
    
control_associations = tmp_df.set_index("RBP KD").to_dict()["Control Accession"]


############################################
# Get the rMATS file #
############################################
rmats_file = [file for file in glob.glob("{}/{}-*{}*/SE.*".format(rmats_data_path, rbp, cell_line)) if "Transfection" not in file and rbp in selected_rbps]
assert len(rmats_file)==1, print(rmats_file)

rmats_file = rmats_file[0]

# paranoia check: the cell line should be the same as what's labeled on the folder 
assert rmats_file.split("/")[-2].split("-")[2] == cell_line

# read rMATS file and subset for relevant columns
rmats_df = pd.read_csv(rmats_file, sep="\t")


############################################
# Add RBP and sample-specific PSI and counts columns #
############################################
rmats_df["RBP_KD_Target"] = rbp

rmats_df["PSI_KD-1"] = rmats_df["IncLevel1"].str.split(",").str[0]
rmats_df["PSI_KD-2"] = rmats_df["IncLevel1"].str.split(",").str[1]

rmats_df["PSI_CTRL-1"] = rmats_df["IncLevel2"].str.split(",").str[0]
rmats_df["PSI_CTRL-2"] = rmats_df["IncLevel2"].str.split(",").str[1]

rmats_df = rmats_df.drop(columns = ["IncLevel1", "IncLevel2"])

rmats_df["Counts_KD-1"] = (rmats_df["IJC_SAMPLE_1"].str.split(",").str[0]).astype("int64") + (rmats_df["SJC_SAMPLE_1"].str.split(",").str[0]).astype("int64")
rmats_df["Counts_KD-2"] = (rmats_df["IJC_SAMPLE_1"].str.split(",").str[1]).astype("int64") + (rmats_df["SJC_SAMPLE_1"].str.split(",").str[1]).astype("int64")
rmats_df["Counts_CTRL-1"] = (rmats_df["IJC_SAMPLE_2"].str.split(",").str[0]).astype("int64") + (rmats_df["SJC_SAMPLE_2"].str.split(",").str[0]).astype("int64")
rmats_df["Counts_CTRL-2"] = (rmats_df["IJC_SAMPLE_2"].str.split(",").str[1]).astype("int64") + (rmats_df["SJC_SAMPLE_2"].str.split(",").str[1]).astype("int64")

rmats_df =  rmats_df.to_dict(orient="records")


############################################
# Get the mapping from splice junction to number of peaks #
############################################
junction_to_num_peaks = {}

with gzip.GzipFile(f"../../3_assign_eCLIP_to_splice_junctions/output/splice_junction_rbp_num_peaks/{cell_line}_{threshold}.pkl.gz", 'rb') as in_file: 
    junction_to_num_peaks = pickle.load(in_file)
        
        
############################################
# CRUX OF THE SCRIPT #
# Create the ML input data #
############################################
ML_input_data = {}
events_encountered = set()
all_zero_events = 0 
kd_binding_present=0

for row in rmats_df: 
    position_definition = exon_ordering[row["strand"]]
    
    for sample in sample_names: 

        if row["PSI_"+sample] != "NA": 

            unique_id = ""
            
            unique_id = "_".join(
                [str(row[coord_column]) for coord_column in rmats_file_column_subset[0:8]]
            )
            
            if "KD" in sample: 
                kd_ctrl_string = row["RBP_KD_Target"] 
                associated_experiment = control_associations[row["RBP_KD_Target"]]
            elif "CTRL" in sample: 
                kd_ctrl_string = control_associations[row["RBP_KD_Target"]]
                associated_experiment = row["RBP_KD_Target"]
            
            unique_id = "_".join(
                [unique_id, kd_ctrl_string, sample]
            )
                                
            if unique_id not in events_encountered: 
                                
                events_encountered.add(unique_id)
                
                binding_present=False
                binding_dict = {}
                
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
                            tmp_num_peaks = junction_to_num_peaks[splice_junction_id][rbp]
                            binding_dict[feature_string] = tmp_num_peaks

                            if tmp_num_peaks > 0: 
                                binding_present=True

                        elif not junction_present: 
                            binding_dict[feature_string] = 0
                
                if binding_present: 

                    ML_input_data[unique_id] = copy.deepcopy(binding_dict)
                    ML_input_data[unique_id]["chr"] = row["chr"]
                                            
                    if "KD" in sample: 
                        tmp_kd_target = row["RBP_KD_Target"]
                        ML_input_data[unique_id]["RBP_KD_Target"] = row["RBP_KD_Target"]

                        for i in range(1,7): 
                            feature_string = "_".join([tmp_kd_target, str(i), "binding"])
                            if ML_input_data[unique_id][feature_string] > 0: 
                                kd_binding_present+=1

                    elif "CTRL" in sample: 
                        ML_input_data[unique_id]["RBP_KD_Target"] = "CTRL"    

                    ML_input_data[unique_id]["Sample Name"] = "_".join(
                        [kd_ctrl_string, sample]
                    )                    
                    
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

                    ML_input_data[unique_id]["rMATS Event ID"] = row["ID"]
                    ML_input_data[unique_id]["ENSEMBL Gene ID"] = row["GeneID"]
                    ML_input_data[unique_id]["Gene Name"] = row["geneSymbol"]
                    ML_input_data[unique_id]["Associated Experiment"] = associated_experiment
                    ML_input_data[unique_id]["Inclusion Isoform Length"] = row["IncFormLen"]
                    ML_input_data[unique_id]["Skipping Isoform Length"] = row["SkipFormLen"]

                    ML_input_data[unique_id]["Raw P-Val"] = row["PValue"]
                    ML_input_data[unique_id]["FDR"] = row["FDR"]
                    ML_input_data[unique_id]["DeltaPSI"] = row["IncLevelDifference"]

                else: 
                    all_zero_events+=1

print(
    [{"all_zero_events": all_zero_events, "total_events": len(ML_input_data), "kd_binding_present": kd_binding_present}]
)


ML_input_data = pd.DataFrame.from_dict(ML_input_data, orient="index")

ML_input_data.to_csv(
    "../output/{}_{}_{}_num-peaks-no-kd.tsv.gz".format(cell_line, args.rbp, threshold), 
    sep="\t",
    compression="gzip"
)


# def retrieve_data_combo(ML_input_data, flavor, expression, normalization_method, log):

#     tmp_output_df = copy.deepcopy(ML_input_data)
    
#     for unique_id in tmp_output_df: 
        
#         row = tmp_output_df[unique_id]
        
#         if flavor=="num-peaks" or flavor=="binary-binding": 

#             if row["RBP_KD_Target"]!="CTRL": 
#                 kd_rbp = row["RBP_KD_Target"]
                
#                 for position in range(1,7): 
#                     feature_string = "_".join([kd_rbp, str(position), "binding"])
#                     row[feature_string] = 0
            
#             if flavor=="binary-binding":
#                 for feature in row: 
#                     if "_binding" in feature and row[feature] >1 :
#                         row[feature] = 1
        
#         elif flavor=="expression-no-num-peaks" or flavor=="expression-yes-num-peaks":
            
#             sample = "_".join(unique_id.split("_")[-2:])

#             for feature in row: 
#                 if "_binding" in feature and row[feature] > 0:
                    
#                     if log:
#                         expression_value = numpy.log2((expression[sample][feature.split("_")[0]]) + 1) 
#                     elif not log: 
#                         expression_value = expression[sample][feature.split("_")[0]]              
                    
#                     if flavor=="expression-no-num-peaks":
#                         row[feature] = expression_value
#                     elif flavor=="expression-yes-num-peaks":
#                         row[feature] = (row[feature]) * (expression_value)

#     tmp_output_df = pd.DataFrame.from_dict(tmp_output_df, orient="index")

#     tmp_output_df.to_csv(
#         "../output/{}_{}_{}_{}-only.tsv.gz".format(cell_line, threshold, flavor, expression, log), 
#         sep="\t",
#         compression="gzip"
#     )


# for flavor in ["expression-no-num-peaks", "expression-yes-num-peaks"]: 
#     for file in glob.glob("../../2_normalize_raw_counts_matrices/outputs/*{}*.tsv.gz".format(cell_line)): 

#         expression = pd.read_csv(file, sep="\t", compression="gzip", index_col=0).to_dict()
#         normalization_method = file.split("/")[-1].split(".")[0].split("_")[1]

#         for log in [True, False]: 
#             retrieve_data_combo(ML_input_data, flavor, expression, normalization_method, log)

# for flavor in ["num-peaks", "binary-binding"]: 
#     retrieve_data_combo(ML_input_data, flavor, None, None, None)

