import polars as pl, pandas as pd, gzip, glob, sys, argparse, os
from loguru import logger
from dataclasses import dataclass

@dataclass
class YogiRbpMlDataValidatorAndExonAdder:

    cell_line: str
    distance: int

    
    def __post_init__(self):

        logger.remove()
        logger.add(sys.stdout, level="INFO")
        logger.add(sys.stderr, level="ERROR")
        logger.info(f"Creating YogiRbpMlDataValidatorAndExonAdder object for {self.cell_line} and {self.distance}")

        self.read_gtf()
        self.read_data()
        self.check_coordinate_ordering()
        self.add_exon_columns()
        self.validate_binding_graphs()
        self.output_data_and_compress()


    def read_gtf(self): 

        gtf = pd.read_csv("/project/PlatigLab/data/annotations/gencode.v29.primary_assembly.annotation_UCSC_names.gtf.gz", sep="\t", compression="gzip", skiprows = 5, header=None)
        gtf = gtf[(gtf[2]=="exon")]

        gtf[3] = gtf[3].astype(int) - 1

        gtf["ENSE"] = gtf[8].str.split('exon_id "').str[1].str.split('"').str[0].str.split(".").str[0]
        gtf["index"] = gtf[0].astype(str) + "_" + gtf[6].astype(str) + "_" + gtf[3].astype(str) + "_" + gtf[4].astype(str)

        gtf = gtf[["ENSE", "index"]].drop_duplicates()

        # Group by "index" and take the first "ENSE" value for each unique "index"
        gtf = gtf.groupby("index").first().reset_index()
        # Convert the DataFrame to a dictionary
        gtf = gtf.set_index("index")["ENSE"].to_dict()

        self.gtf = gtf

    
    def read_data(self,): 

        file = glob.glob(f"../final_modeling_input_datasets/{self.cell_line}_{self.distance}_*.gz")
        assert len(file)==1, f"More than one file found for {self.cell_line} and {self.distance}"

        # Load the DataFrame
        df = pl.read_csv(file[0], has_header=True, separator="\t")
        # Rename the first column to "index"
        df = df.rename({df.columns[0]: "index"})

        self.df = df 


    def check_coordinate_ordering(self):
            
        # Iterate over rows and assert the order of elements
        for i in range(len(self.df)):
            index_split = self.df[i, "index"].split("_")

            strand = index_split[1]
            elements = list(map(int, index_split[2:8]))

            if strand=="+":
                assert elements == sorted(elements), logger.error(f"Values are not in order for index {self.df[i, 'index']}")
            elif strand=="-":
                assert elements == sorted(elements, reverse=True), logger.error(f"Values are not in order for index {self.df[i, 'index']}")

        logger.info("All coordinates are in expected order")


    def add_exon_columns(self): 

        # Initialize dictionaries to store exon ids
        exon_id_dict = {
            "Upstream Exon": [],
            "Main Exon": [],
            "Downstream Exon": []
        }

        # Iterate over rows and fill new columns
        for i in range(len(self.df)):
            index_split = self.df[i, "index"].split("_")

            chromosome = index_split[0]
            strand = index_split[1]

            if strand == "+":
                upstream_exon = self.gtf["{}_{}_{}_{}".format(chromosome, strand, index_split[2], index_split[3])]
                main_exon = self.gtf["{}_{}_{}_{}".format(chromosome, strand, index_split[4], index_split[5])]
                downstream_exon = self.gtf["{}_{}_{}_{}".format(chromosome, strand, index_split[6], index_split[7])]
            
            elif strand == "-":
                upstream_exon = self.gtf["{}_{}_{}_{}".format(chromosome, strand, index_split[3], index_split[2])]
                main_exon = self.gtf["{}_{}_{}_{}".format(chromosome, strand, index_split[5], index_split[4])]
                downstream_exon = self.gtf["{}_{}_{}_{}".format(chromosome, strand, index_split[7], index_split[6])]
            
            exon_id_dict["Upstream Exon"].append(upstream_exon)
            exon_id_dict["Main Exon"].append(main_exon)
            exon_id_dict["Downstream Exon"].append(downstream_exon)            

        self.df = self.df.with_columns([
            pl.Series("Upstream Exon", exon_id_dict["Upstream Exon"]),
            pl.Series("Main Exon", exon_id_dict["Main Exon"]),
            pl.Series("Downstream Exon", exon_id_dict["Downstream Exon"])
        ])

        # Assert that all values in new columns begin with ENSE
        all_start_with_ense = (
            self.df["Upstream Exon"].str.starts_with("ENSE") &
            self.df["Main Exon"].str.starts_with("ENSE") &
            self.df["Downstream Exon"].str.starts_with("ENSE")
        )
        assert all_start_with_ense.sum() == len(self.df), "Not all values in exon columns start with 'ENSE'"

        logger.success("Exon columns added successfully")

    
    def validate_binding_graphs(self):
        # Group by "Upstream Exon", "Main Exon", "Downstream Exon"
        grouped = self.df.group_by(["Upstream Exon", "Main Exon", "Downstream Exon"])

        for group, data in grouped:

            # Select columns that end with "_binding"
            binding_columns = [col for col in data.columns if col.endswith("_binding")]
            binding_df = data.select(binding_columns)

            # Check if all rows are duplicates of each other
            assert binding_df.n_unique() == 1, logger.error(f"Rows are not duplicates for combination {group}")
        
        logger.info("All 3-exon combinations have the same binding graph")

    
    def output_data_and_compress(self):

        logger.info("Outputting data to CSV file and GZIP compressing")

        OUTPUT_DIR="/project/PlatigLab/data/RBP_ML/3_yogi_dataset_january_2025/all_events/"

        # Output the DataFrame to a CSV file
        self.df.write_csv(f"{OUTPUT_DIR}/{self.cell_line}_{self.distance}_num-peaks-no-kd.tsv", separator="\t")

        # Compress the CSV file
        os.system(f"gzip {OUTPUT_DIR}/{self.cell_line}_{self.distance}_num-peaks-no-kd.tsv")

        logger.success("Data outputted and compressed successfully")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create exon assignment and run data assertions for Yogi RBP ML data")
    parser.add_argument("--parallelize", action="store_true", help="Flag to parallelize the process")
    parser.add_argument("--cell_line", type=str, help="Specify the cell line")
    parser.add_argument("--distance", type=int, help="Specify the distance")

    args = parser.parse_args()

    if args.parallelize:
        assert not args.cell_line and not args.distance, "Both cell_line and distance arguments must be provided"

        cell_lines = ["K562", "HepG2",]
        distances = ["25", "50", "75", "100", "150", "200", "250", "500", "1000"]

        for cell_line in cell_lines: 
            for distance in distances:

                output_file = f"/project/PlatigLab/data/RBP_ML/3_yogi_dataset_january_2025/all_events/{cell_line}_{distance}_num-peaks-no-kd.tsv.gz"
                
                if not os.path.exists(output_file):
                    os.system(
                        f"sbatch --partition=standard --account=platiglab -N1 -n7 --mem=200GB --output=../SLURM_output/exon_assignment_and_data_assertions_{cell_line}_{distance}.out --error=../SLURM_output/exon_assignment_and_data_assertions_{cell_line}_{distance}.err --wrap='python3.11 ./4_exon_assignment_and_data_assertions.py --cell_line {cell_line} --distance {distance}'"
                    )
        
    else: 
        assert args.cell_line and args.distance, "Both cell_line and distance arguments must be provided"
        YogiRbpMlDataValidatorAndExonAdder(args.cell_line, args.distance)



