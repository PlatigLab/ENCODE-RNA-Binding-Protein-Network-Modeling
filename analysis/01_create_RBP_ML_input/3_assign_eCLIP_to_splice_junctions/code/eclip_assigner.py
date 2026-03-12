from dataclasses import dataclass
from loguru import logger
import polars as pl, pandas as pd
import glob, os, sys, gzip, pickle

pl.Config.set_tbl_cols(-1)  # Show all columns
pl.Config.set_tbl_rows(-1)  # Show all rows
pl.Config.set_tbl_width_chars(10000)  # Set column width

@dataclass
class EclipToSpliceJunctionAssigner:

    cell_lines = ["K562", "HepG2"]

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

    # columns with coordinate numbers 
    exon_coordinate_columns = ["upstreamES", "upstreamEE", "exonStart_0base", "exonEnd", "downstreamES", "downstreamEE"]
    # unique id information 
    id_creation_columns = ["chr", "start", "strand"]
    # final order of columns for BED file
    final_bed_order = ["chr", "start", "end", "name", "score", "strand"]
    # glob path to all SE files 
    SE_glob = "/project/PlatigLab/data/collaborators/BWH/1_ENCODE_shRNA_RBP_KD_2024-04-hg38-gencode-v29/**/SE*.txt"

    # distance thresholds for assigning eCLIP peaks to splice junctions, in base pairs
    thresholds = [25, 50, 75, 100, 150, 200, 250, 500, 1000]
    # only consider autosomal chromosomes
    allowed_chromosomes = [f"chr{i}" for i in list(range(1, 23))]

    # for parsing exons from rMATS events and making sure they are in allowed exons set
    exon_definition_columns = (
        ["upstreamES", "upstreamEE"], 
        ["exonStart_0base", "exonEnd"], 
        ["downstreamES", "downstreamEE"]
    )

    def __post_init__(self):
        logger.remove()
        logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")
        
        self.get_eCLIP_and_shRNA_RBPs()
        self.retrieve_gencode_v24_and_v29_overlapping_exons()
        self.get_events()
        self.get_all_splice_junctions()
        self.get_eCLIP_peaks()
        self.run_bedtools_mapping()
        self.create_splice_junction_to_num_peak_dict()


    def get_eCLIP_and_shRNA_RBPs(self):

        eclip_all_rbps = {}
        shrna_all_rbps = {}

        shrna_metadata = pd.read_csv(
            "../../../../inputs/metadata/ENCODE_shRNA_knockdown_FASTQ_metadata.tsv",
            sep="\t",
        )  
        

        eclip_metadata = pd.read_csv(
            "../../../../inputs/metadata/eclip_metadata.tsv", 
            sep="\t"
        )   

        shrna_metadata = shrna_metadata[
            (shrna_metadata["Library made from"]=="polyadenylated mRNA") & 
            (shrna_metadata["File Status"]=="released") 
        ]

        # subset eCLIP metadata to hg38 genome, released files, and IDR peaks
        eclip_metadata = eclip_metadata[
            (eclip_metadata["File assembly"]=="GRCh38") & 
            (eclip_metadata["File Status"]=="released") &
            (eclip_metadata["Biological replicate(s)"]=="1, 2") 
        ]

        # for each cell line 
        for cell_line in self.cell_lines: 

            # get all RBPs for shRNA data that match that cell line 
            shrna_rbps = set(shrna_metadata[shrna_metadata["Biosample term name"]==cell_line]["Experiment target"].str.split("-").str[0].to_list())
            # get all RBPs for eCLIP in that cell line 
            eclip_rbps = set(eclip_metadata[eclip_metadata["Biosample term name"]==cell_line]["Experiment target"].str.split("-").str[0].to_list())

            # get sorted list of RBPs in both shRNA and eCLIP for that cell line
            shrna_all_rbps[cell_line] = sorted(list(eclip_rbps.intersection(shrna_rbps)))
            # get sorted list of RBPs with eCLIP data for that cell line
            eclip_all_rbps[cell_line] = sorted(list(eclip_rbps))

            logger.info(f"Number of eCLIP RBPs for {cell_line}: {len(eclip_all_rbps[cell_line])}")
            logger.info(f"Number of shRNA RBPs for {cell_line}: {len(shrna_all_rbps[cell_line])}")

            # Output eCLIP RBPs to a file
            eclip_output_file = f"../output/eclip_and_shrna_rbps/{cell_line}_eclip_rbps.txt"
            with open(eclip_output_file, 'w') as f:
                for rbp in eclip_all_rbps[cell_line]:
                    f.write(f"{rbp}\n")

            # Output shRNA RBPs to a file
            shrna_output_file = f"../output/eclip_and_shrna_rbps/{cell_line}_shrna_rbps.txt"
            with open(shrna_output_file, 'w') as f:
                for rbp in shrna_all_rbps[cell_line]:
                    f.write(f"{rbp}\n")

        self.eclip_all_rbps = eclip_all_rbps
        self.shrna_all_rbps = shrna_all_rbps

        self.eclip_metadata = eclip_metadata
        
    
    def retrieve_gencode_v24_and_v29_overlapping_exons(self): 
        
        # Load all gunzipped GTF files from the annotations directory
        gtf_files = [file for file in glob.glob("../../../../inputs/annotations/*.gtf.gz") if ".v24." in file or ".v29." in file]
        assert len(gtf_files) == 2, f"Expected 2 GTF files (v24 and v29), but found {len(gtf_files)}: {gtf_files}"  

        # Read and process each GTF file separately
        exon_sets = []
        for gtf_file in gtf_files:
            # Read GTF file and filter for exon features
            gtf_data = pl.scan_csv(
                gtf_file,
                separator="\t",
                has_header=False,
                comment_prefix="#",
                new_columns=["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
            ).filter(
                (pl.col("feature") == "exon") & 
                (pl.col("seqname").is_in(self.allowed_chromosomes))
            ).select(
                ["seqname", "start", "end", "strand"]
            ).collect()

            # Convert to 0-based BED-style coordinates (subtract 1 from start)
            gtf_data = gtf_data.with_columns(
                (pl.col("start") - 1).alias("start_0based")
            ).select(
                [pl.col("seqname"), pl.col("start_0based"), pl.col("end"), pl.col("strand")]
            )

            # Create exon IDs - same concatenation order for all strands
            exon_ids = gtf_data.with_columns(
                pl.concat_str(
                    [pl.col("seqname").cast(pl.Utf8), pl.col("strand").cast(pl.Utf8), pl.col("start_0based").cast(pl.Utf8), pl.col("end").cast(pl.Utf8)],
                    separator="_"
                ).alias("exon_id")
            ).select("exon_id")
            
            exon_sets.append(set(exon_ids.to_series().to_list()))

        matching_exons = exon_sets[0].intersection(exon_sets[1])
        logger.info(f"Number of valid exons present in both Gencode v24 and v29: {len(matching_exons)}")

        matching_exons_sorted = sorted(list(matching_exons))
        output_file = "../output/allowed_exons/matching_exons.txt"
        with open(output_file, 'w') as f:
            for exon in matching_exons_sorted:
                f.write(f"{exon}\n")
        
        self.annotation_matching_exons = matching_exons


    def get_events(self):

        all_events = {}
        non_overlapping_events = {}

        for cell_line in self.cell_lines:

            # get paths to all SE event rMATS files
            SE_files = glob.glob(
                self.SE_glob, 
                recursive=True
            )
            SE_files = [
                file for file in SE_files 
                if "-Transfection-" not in file 
                and f"-{cell_line}/" in file
                and file.split("/")[-2].split("-")[0] in self.shrna_all_rbps[cell_line]
            ]

            logger.info(f"Number of {cell_line} Skipped Exon (SE) files: {len(SE_files)}")

            dataframe = pl.scan_csv(
                SE_files, 
                separator="\t",  
            ).select(
                ['chr', 'strand'] + self.exon_coordinate_columns
            ).unique().sort(
                ['chr', 'strand', self.exon_coordinate_columns[0]]
            ).collect(streaming=True)

            dataframe = dataframe.filter(
                pl.col("chr").is_in(self.allowed_chromosomes)
            )

            # Create a boolean column for each exon pair, then filter rows where ALL exon pairs are valid
            for idx, exon_def_cols in enumerate(self.exon_definition_columns, 1):
                dataframe = dataframe.with_columns(
                    pl.concat_str(
                        [pl.col("chr").cast(pl.Utf8), pl.col("strand").cast(pl.Utf8)] + \
                            [pl.col(col).cast(pl.Utf8) for col in exon_def_cols], 
                        separator="_"
                    ).is_in(self.annotation_matching_exons).alias(f"valid_exon_{idx}")
                )
            
            # Filter to keep only rows where all exon pairs are valid
            valid_columns = [col for col in dataframe.columns if col.startswith("valid_")]
            dataframe = dataframe.filter(
                pl.all_horizontal(valid_columns)
            ).drop(valid_columns)

            logger.info(f"Number of unique {cell_line} events: {dataframe.shape[0]}")

            # Assert that for each row, the exon coordinate columns are in increasing order
            for row in dataframe.iter_rows(named=False):
                coordinates = [row[i] for i in range(2, 8)]  # indices corresponding to exon_coordinate_columns
                assert coordinates == sorted(coordinates), "Exon coordinate columns are not in increasing order"

            non_overlapping = []
            grouped = dataframe.group_by(["chr", "strand"], maintain_order=True)

            for group, df in grouped:
                df = df.sort(by=self.exon_coordinate_columns[0])
                highest_value = -1

                for row in df.iter_rows(named=False):
                    if row[2] > highest_value:  # index 2 corresponds to the first exon coordinate column
                        non_overlapping.append(row)
                        highest_value = row[7]  # index 7 corresponds to the last exon coordinate column

            non_overlapping = pl.DataFrame(
                non_overlapping, 
                schema={"chr": pl.Utf8, "strand": pl.Utf8, **{col: pl.Int64 for col in self.exon_coordinate_columns}}, 
                orient='row'
            )

            for df_name, df in [("non-overlapping", non_overlapping), ("all-events", dataframe)]:
                # Separate the dataframe into + and - strand
                pos_strand_df = df.filter(pl.col("strand") == "+")
                neg_strand_df = df.filter(pl.col("strand") == "-")

                # Concatenate all columns from left to right to create event_id
                pos_strand_df = pos_strand_df.with_columns(
                    pl.concat_str(
                        [
                            pl.col(col).cast(pl.Utf8) for col in ["chr", "strand"] + self.exon_coordinate_columns
                        ], 
                        separator="_"
                    ).alias("event_id")
                )

                neg_strand_df = neg_strand_df.with_columns(
                    pl.concat_str(
                        [
                            pl.col(col).cast(pl.Utf8) for col in ["chr", "strand"] + list(reversed(self.exon_coordinate_columns))
                        ], 
                        separator="_"
                    ).alias("event_id")
                )
                
                # Concatenate the event_id for the + and - strand dataframes
                combined_df = pl.concat([pos_strand_df.select("event_id"), neg_strand_df.select("event_id")])

                # Output the combined event_id dataframe to a file
                output_file = f"../output/SE_events/{cell_line}_{df_name}.tsv"
                combined_df.write_csv(output_file, separator="\t", include_header=False)

                if df_name == "non-overlapping":
                    non_overlapping_events[cell_line] = combined_df
                else:
                    all_events[cell_line] = combined_df

        self.non_overlapping_events = non_overlapping_events
        self.all_events = all_events
        

    def get_all_splice_junctions(self):

        for cell_line in self.cell_lines: 
            for type, dataframe in [('all-events', self.all_events), ('non-overlapping', self.non_overlapping_events)]:
                
                event_ids = dataframe[cell_line].to_series().to_list()
                splice_junctions = []

                for event_id in event_ids:
                    parts = event_id.split("_")
                    chromosome = parts[0]
                    strand = parts[1]
                    coordinates = parts[2:]

                    for coord in coordinates:
                        start = int(coord)
                        end = start + 1
                        name = f"{chromosome}_{strand}_{coord}"
                        splice_junctions.append([chromosome, start, end, name, ".", strand])

                splice_junctions_df = pl.DataFrame(
                    splice_junctions, 
                    schema={
                        "chr": pl.Utf8, 
                        "start": pl.Int64, 
                        "end": pl.Int64, 
                        "name": pl.Utf8, 
                        "score": pl.Utf8, 
                        "strand": pl.Utf8
                    },
                    orient='row'  # Explicitly mention that the input is in row form
                )

                splice_junctions_df = splice_junctions_df.unique()
                assert splice_junctions_df["name"].n_unique() == splice_junctions_df.shape[0], "The 'name' column contains duplicate values"

                logger.info(f"Number of splice junctions for {type} {cell_line}: {splice_junctions_df.shape[0]}")

                output_file = f"../output/bedtools_input/{cell_line}_{type}_splice_junctions.bed"
                splice_junctions_df.write_csv(output_file, separator="\t", include_header=False)

    
    def get_eCLIP_peaks(self): 
        # key is cell line and value is all peaks 
        # for RBPs with eCLIP in that cell line
        all_peaks = {}

        # for each cell line 
        for cell_line in self.cell_lines: 
            
            # list of dataframes that will be concatenated to 
            # get all RBPs peaks in a given cell line 
            concat_peaks = []
            
            # for each RBP in both assays 
            for rbp in self.eclip_all_rbps[cell_line]: 
                # subset eCLIP to cell line and RBP 
                tmp_metadata = self.eclip_metadata[
                    (self.eclip_metadata["Biosample term name"]==cell_line) & 
                    (self.eclip_metadata["Experiment target"]==f"{rbp}-human")
                ]
                
                # for the few cases where they re-did the eCLIP
                if tmp_metadata.index.size != 1:
                    logger.info(f"{rbp} has {tmp_metadata.index.size} eCLIP experiments")
                    # get the newest eCLIP experiment
                    tmp_metadata = tmp_metadata.sort_values("Experiment date released", ascending=False).head(n=1)
                                    
                assert tmp_metadata.index.size == 1, tmp_metadata
                
                # get the file associated with that cell line and RBP 
                file = glob.glob(
                    f"/project/PlatigLab/data/ENCORE/eCLIP/encode/peaks/{tmp_metadata.iloc[0, 0]}.bed.gz"
                )
                assert len(file) == 1
                
                # load eCLIP peaks file 
                tmp_peaks = pl.read_csv(
                    file[0], 
                    has_header=False, 
                    separator="\t", 
                    columns=[0, 1, 2, 3, 4, 5]
                )     
                
                # since we are concatenating all peaks together downstream
                # make sure to identify peaks by their rbp_cell-line_filtering methods
                tmp_peaks = tmp_peaks.with_columns(
                    pl.lit(rbp + "_" + cell_line + "_IDR").alias("column_4")
                ).select(
                    [pl.col("column_1"), pl.col("column_2"), pl.col("column_3"), pl.col("column_4"), pl.col("column_5"), pl.col("column_6")]
                )
                
                # add this rbps peaks to the list of peaks for the cell line 
                concat_peaks.append(tmp_peaks)
    
            # concatenate all the peaks for the whole cell line 
            all_peaks[cell_line] = pl.concat(concat_peaks).sort(["column_1", "column_2",])
            assert all_peaks[cell_line].shape[0] == all_peaks[cell_line].unique().shape[0]

            # Output the concatenated peaks to a file
            output_file = f"../output/bedtools_input/{cell_line}_eclip_peaks.bed"
            all_peaks[cell_line].write_csv(output_file, separator="\t", include_header=False)

        self.all_peaks = all_peaks


    def run_bedtools_mapping(self): 

        bedtools_path="/project/PlatigLab/software/bedtools-v2.31.1/bin/bedtools"
        
        for cell_line in self.cell_lines: 

            # get the eCLIP peaks for that cell line 
            eclip_peaks = f"../output/bedtools_input/{cell_line}_eclip_peaks.bed"
            sorted_eclip_peaks = f"../output/bedtools_input/{cell_line}_eclip_peaks_sorted.bed"

            result = os.system(f"{bedtools_path} sort -i {eclip_peaks} > {sorted_eclip_peaks}")
            assert result == 0, f"bedtools sort failed for {eclip_peaks} with return code {result}"

            for type in ['all-events', 'non-overlapping']: 
                # get the splice junctions for that cell line and type 
                splice_junctions = f"../output/bedtools_input/{cell_line}_{type}_splice_junctions.bed"
                # Sort the splice junctions and eCLIP peaks files using bedtools sort
                sorted_splice_junctions = f"../output/bedtools_input/{cell_line}_{type}_splice_junctions_sorted.bed"

                result = os.system(f"{bedtools_path} sort -i {splice_junctions} > {sorted_splice_junctions}")
                assert result == 0, f"bedtools sort failed for {splice_junctions} with return code {result}"

                # get the output file 
                output_file = f"../output/peaks_to_splice_junctions/{cell_line}_{type}_peaks_to_splice_junctions.bed"
                # run bedtools intersect 
                result = os.system(f"{bedtools_path} closest -a {sorted_eclip_peaks} -b {sorted_splice_junctions} -s -d -t all > {output_file}")
                assert result == 0, f"bedtools intersect failed for {sorted_eclip_peaks} and {sorted_splice_junctions} with return code {result}"

                os.remove(splice_junctions)

            os.remove(eclip_peaks)

    
    def create_splice_junction_to_num_peak_dict(self): 

        splice_junction_to_num_peak = {}

        for cell_line in self.cell_lines: 
            splice_junction_to_num_peak[cell_line] = {}

            for type in ['all-events', 'non-overlapping']: 
                splice_junction_to_num_peak[cell_line][type] = {}

                # get the output file 
                input_file = f"../output/peaks_to_splice_junctions/{cell_line}_{type}_peaks_to_splice_junctions.bed"
                # read in the output file 
                peaks_to_splice_junctions = pl.read_csv(
                    input_file, 
                    separator="\t", 
                    has_header=False
                )
                peaks_to_splice_junctions.columns = [str(i) for i in range(peaks_to_splice_junctions.width)]

                logger.info(f"{cell_line} {type} initial shape: {peaks_to_splice_junctions.shape[0]}")

                # Get the distance column (last column in bed file)
                distance_column = peaks_to_splice_junctions.columns[-1]
                # Remove the peaks that did not have any splice junctions to map to
                peaks_to_splice_junctions = peaks_to_splice_junctions.filter(pl.col(distance_column) >= 0)
                
                assert peaks_to_splice_junctions.null_count().sum_horizontal().item()==0, "Null values found in the dataframe after filtering for valid mappings"
                # Assert that the 6th and 12th columns are equal in the whole dataframe
                assert (peaks_to_splice_junctions[:, 5] == peaks_to_splice_junctions[:, 11]).all(), "6th and 12th columns are not equal"

                logger.info(f"After filtering out missing mappings for {cell_line} {type}: {peaks_to_splice_junctions.shape[0]}")

                peaks_to_splice_junctions = peaks_to_splice_junctions.sample(
                    fraction=1, seed=17
                ).unique(
                    subset=["0", "1", "2", "3", "4", "5"], 
                    keep='first', 
                    maintain_order=True
                )
                peaks_to_splice_junctions = peaks_to_splice_junctions.sort(by=["0", "1", "2", "3", "4", "5"])

                ties = peaks_to_splice_junctions.group_by(["0", "1", "2", "3", "4", "5"]).len()
                assert (ties["len"] == 1).all(), "Duplicate eCLIP assignments found"

                logger.info(f"After removing tied CLIP assignments to create unique binding only for {cell_line} {type}: {peaks_to_splice_junctions.shape[0]}")

                # Filter for the RBP, unique splice junction ID, and distance columns
                peaks_to_splice_junctions = peaks_to_splice_junctions.select(["3", "9", distance_column])
                # Parse out RBP column
                peaks_to_splice_junctions = peaks_to_splice_junctions.with_columns(
                    pl.col('3').str.split("_").list.get(0).alias("RBP")
                )
                # Change column names for clarity
                peaks_to_splice_junctions = peaks_to_splice_junctions.rename(
                    {
                        '9': "Splice Junction ID",
                        distance_column: "Distance"
                    }
                )
                
                # Ensure there are no null values in matrix
                assert peaks_to_splice_junctions.null_count().sum_horizontal().sum()==0, "Null values found in the dataframe"

                # for each distance threshold 
                for threshold in self.thresholds: 
                    
                    # subset to less than distance 
                    threshold_mappings = peaks_to_splice_junctions.filter(pl.col("Distance") <= threshold)

                    threshold_mappings = threshold_mappings.group_by(
                        ["Splice Junction ID", "RBP"]
                    ).len().sort(
                        ["Splice Junction ID", "RBP"]
                    ).pivot(
                        values="len", 
                        index="Splice Junction ID", 
                        on="RBP", 
                        maintain_order=True, 
                        sort_columns=True,
                        aggregate_function=None
                    ).fill_null(0)

                    # get the rbps that should be included in the dataset but had no data show up 
                    missing_rbps = set(self.eclip_all_rbps[cell_line]) - set(threshold_mappings.columns)
                        
                    # if there are missing rbps 
                    if len(missing_rbps) > 0: 
                        logger.info(f"Missing RBPs for {cell_line} at threshold {threshold}: {', '.join(sorted(missing_rbps))}")

                        # fill all values for the rbp as 0 
                        for rbp in missing_rbps: 
                            threshold_mappings = threshold_mappings.with_columns(pl.lit(0).alias(rbp))
                    
                    # Convert the polars dataframe to pandas
                    threshold_mappings = threshold_mappings.to_pandas()
                    
                    # Set the splice junction id as the index
                    threshold_mappings = threshold_mappings.set_index("Splice Junction ID")
                    
                    # Sort both the index and the columns
                    threshold_mappings = threshold_mappings.sort_index()
                    threshold_mappings = threshold_mappings[sorted(threshold_mappings.columns)]

                    # Assert that the maximum value in the entire dataframe is greater than 1
                    assert threshold_mappings.max().max() > 1
                    # Assert that the minimum value in the entire dataframe is 0
                    assert threshold_mappings.min().min() == 0, "Minimum value in the dataframe is not 0"
                    
                    # Output to dict using orient='index'
                    threshold_mappings_dict = threshold_mappings.to_dict(orient='index')

                    assert threshold not in splice_junction_to_num_peak[cell_line][type], "Threshold already present in the dictionary"
                    splice_junction_to_num_peak[cell_line][type][threshold] = threshold_mappings_dict

            # Output the dictionary to a gzip file using pickle
            output_file = f"../output/splice_junction_rbp_num_peaks/{cell_line}_RBP_peaks_per_splice_junction.pkl.gz"
            assert len(splice_junction_to_num_peak[cell_line]) == 2
            assert len(splice_junction_to_num_peak[cell_line]['all-events']) == len(splice_junction_to_num_peak[cell_line]['non-overlapping'])

            with gzip.open(output_file, 'wb') as f:
                pickle.dump(splice_junction_to_num_peak[cell_line], f)
        
        self.splice_junction_to_num_peak = splice_junction_to_num_peak

if __name__ == "__main__":
    EclipToSpliceJunctionAssigner()
    logger.success("Finished running eCLIP to splice junction assignment for all cell lines and dataset types.")

    