import pandas as pd
import pybedtools
import os
import glob
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Process TF binding data.')
parser.add_argument('tf1', type=str, help='First transcription factor')
parser.add_argument('tf2', type=str, help='Second transcription factor')

args = parser.parse_args()

tf1 = args.tf1
tf2 = args.tf2

cell_line_mapping = {
    "293": "HEK_293",
    "A549": "A549",
    "HUH-7": "HuH-7",
    "K-562": "K562",
    "MCF-7": "MCF7",
    "22RV1": "22Rv1",
    "A-375": "A-375",
    "C4-2": "C4-2",
    "LNCAP": "LNCAP",
    "PC-3": "PC-3",
    "THP-1": "THP-1",
    "UM-UC-3": "UM-UC-3",
    "VCAP": "VCAP",
    "CAMA-1": "CAMA-1",
    "ECC-1": "AN3_CA",
    "HCC1954": "HCC1954",
    "HEC-1-A": "HEC-1-A",
    "ISHIKAWA": "Ishikawa",
    "MDA-MB-134-VI": "MDA-MB-134-VI",
    "MDA-MB-453": "MDA-MB-453",
    "NCI-H3122": "NCI-H3122",
    "NCI-H441": "NCI-H441",
    "RT4": "RT4",
    "SK-BR-3": "SK-BR-3",
    "GP5D": "GP5D",
    "LS-180": "LS-180",
    "CFPAC-1": "CFPAC-1",
    "GM12878": "GM12878",
    "MDA-MB-231": "MDA-MB-231",
    "HELA": "HELA",
    "SK-N-SH": "SK-N-SH",
    "HMEC": "HMEC",
    "U2OS": "U2OS",
    "BJ": "BJ_hTERT",
    "KELLY": "KELLY",
    "U-87 MG": "U-87_MG",
    "MV-4-11": "MV-4-11",
    "MM.1S": "MM.1S",
    "MOLT-4": "MOLT-4",
    "REH": "REH",
    "Raji": "Raji",
    "Ramos": "Ramos",
    "Hep G2": "Hep_G2",
    "NCI-H660": "NCI-H660",
    "NB69": "NB69",
    "SK-N-AS": "SK-N-AS",
    "HEC-1-B": "HEC-1-B",
    "HT29": "HT29",
    "OCI-LY8": "OCI-LY8",
    "DLD-1": "DLD-1",
    "U-937": "U-937",
    "IMR-90": "IMR-90",
    "SK-MEL-28": "SK-MEL-28",
    "MKN-45": "MKN-45",
    "AGS": "AGS",
    "HCT-15": "HCT-15",
    "SW 620": "SW_620",
    "HT1376": "HT-1376",
    "NALM-6": "NALM-6",
    "NAMALWA": "NAMALWA",
    "DOHH-2": "DOHH-2",
    "OCI-LY1": "OCI-LY1",
    "U-266": "U-266",
    "SUM 159PT": "SUM-159PT",
    "KG-1": "KG-1",
    "COLO 205": "COLO_205",
    "CAL-27": "CAL_27",
    "PANC-1": "PANC-1",
    "DU 145": "DU_145",
    "CAL-51": "CAL_51",
    "OCI-AML3": "OCI-AML3",
    "KARPAS-299": "KARPAS-299",
    "HL-60": "HL-60",
    "RH-41": "RH-41",
    "OVCAR-8": "OVCAR-8",
    "NTERA-2": "NTERA-2",
    "SW 780": "SW_780",
    "RD": "RD",
    "T24": "T24",
    "OVCAR-3": "OVCAR-3",
    "SK-N-FI": "SK-N-FI",
    "JURKAT": "JURKAT",
    "DAUDI": "Daudi",
    "K562": "K562",
    "Raji": "Raji",
    "NALM-6": "NALM-6",
    "NCI-H69": "NCI-H69",
    "RPMI-8226": "RPMI_8226",
    "NCI-H1437": "NCI-H1437",
    "COLO-320": "COLO_320",
    "LP-1": "LP-1",
    "NCI-H929": "NCI-H929",
    "OCIMY-5": "OCI-My5",
    "MOLT4": "MOLT-4",
    "RS411": "RS411",
    "OCI-AML3": "OCI-AML3",
    "OCI-LY3": "OCI-LY3",
    "OPM2": "OPM-2",
    "BEAS-2B": "BEAS-2B",
    "HCC70": "HCC70",
    "Capan-1": "Capan-1",
    "NTERA-2": "NTERA-2",
    "HCT-116": "HCT_116",
    "DU145": "DU145",
    "U-2OS": "U-2OS"
}

# Function to get the folder name from the TF1 file name
def get_folder_name(file_name, tf):
    cell_line = os.path.basename(file_name).replace(f"_{tf}.bed", "")
    return cell_line_mapping.get(cell_line, None)

# Function to filter DataFrame based on given criteria
def filter_df(df, threshold=4000):
    df = df[df['end'] - df['start'] <= threshold]
    if df['count'].max() > 50:
        return df.query('count >= @df["count"].quantile(.75)')
    elif df['count'].max() <= 2:
        return df
    else:
        return df.query('count >= @df["count"].median()')

def filter_chromosomes(df, threshold=4000):
    df = df[df['end'] - df['start'] <= threshold]
    valid_chromosomes = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY']
    return df[df['chr'].isin(valid_chromosomes)]

# Directories containing the BED files
directory_tf1 = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{tf1}"
directory_tf2 = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{tf2}"

# List all TF1 and TF2 files
tf1_files = glob.glob(os.path.join(directory_tf1, f"*_{tf1}.bed"))
tf2_files = glob.glob(os.path.join(directory_tf2, f"*_{tf2}.bed"))

# Create sets of cell lines for TF1 and TF2
tf1_cell_lines = {os.path.basename(f).replace(f"_{tf1}.bed", "") for f in tf1_files}
tf2_cell_lines = {os.path.basename(f).replace(f"_{tf2}.bed", "") for f in tf2_files}

# Find common cell lines
common_cell_lines = tf1_cell_lines & tf2_cell_lines

# List to store all DataFrames
combined_data = []

# Process each pair of TF1 and TF2 files for common cell lines
for cell_line in common_cell_lines:
    tf1_file = os.path.join(directory_tf1, f"{cell_line}_{tf1}.bed")
    tf2_file = os.path.join(directory_tf2, f"{cell_line}_{tf2}.bed")
    
    folder_name = get_folder_name(tf1_file, tf1)

    if folder_name is None:
        continue
    
    # Check if the corresponding TF2 file exists
    if os.path.exists(tf2_file):
        # Read the BED files into DataFrames
        tf1_df = pd.read_csv(tf1_file, sep="\t", header=None, names=['chr', 'start', 'end', 'count'])
        tf2_df = pd.read_csv(tf2_file, sep="\t", header=None, names=['chr', 'start', 'end', 'count'])
        
        negative_filepath = f"/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/{folder_name}/peaks/{folder_name}.filtered.broadPeak"

        neg_df = pd.read_csv(negative_filepath, sep="\t", usecols=[0, 1, 2], header=None, names=['chr', 'start', 'end'])

        # filter the intersections
        tf1_bedtool = pybedtools.BedTool.from_dataframe(tf1_df)
        tf2_bedtool = pybedtools.BedTool.from_dataframe(tf2_df)
        negative_bedtool = pybedtools.BedTool.from_dataframe(neg_df)

        negative_bedtool = negative_bedtool.subtract(tf1_bedtool, A=True)
        negative_bedtool = negative_bedtool.subtract(tf2_bedtool, A=True)
        neg_df = negative_bedtool.to_dataframe()
        neg_df = neg_df.rename(columns={"chrom":"chr",})

        neg_df = filter_chromosomes(neg_df)

        # Add count of 1 to neg_df
        neg_df['count'] = 1

        # Filter the DataFrames
        tf1_filtered = filter_df(tf1_df)
        tf2_filtered = filter_df(tf2_df)

        # Convert the filtered DataFrames to BedTool objects
        tf1_bedtool = pybedtools.BedTool.from_dataframe(tf1_filtered)
        tf2_bedtool = pybedtools.BedTool.from_dataframe(tf2_filtered)

        # Perform the subtraction
        tf1_only = tf1_bedtool.subtract(tf2_bedtool, A=True)
        tf2_only = tf2_bedtool.subtract(tf1_bedtool, A=True)

        # Perform the intersection
        intersection = tf1_bedtool.intersect(tf2_bedtool, u=True)

        # Convert the results back to DataFrames
        tf1_only_df = tf1_only.to_dataframe()
        tf2_only_df = tf2_only.to_dataframe()
        intersection_df = intersection.to_dataframe()

        tf1_only_df = tf1_only_df.rename(columns={"chrom":"chr","name":"count"})
        tf2_only_df = tf2_only_df.rename(columns={"chrom":"chr","name":"count"})
        intersection_df = intersection_df.rename(columns={"chrom":"chr","name":"count"})

        # Add cell line column to each DataFrame
        tf1_only_df['cell_line'] = folder_name
        tf2_only_df['cell_line'] = folder_name
        intersection_df['cell_line'] = folder_name
        neg_df['cell_line'] = folder_name

        # Add label column to each DataFrame
        intersection_df['label'] = 1
        tf1_only_df['label'] = 0
        tf2_only_df['label'] = 0
        neg_df['label'] = 0

        intersection_df['source'] = 'intersect'
        tf1_only_df['source'] = f'{tf1}_only'
        tf2_only_df['source'] = f'{tf2}_only'
        neg_df['source'] = 'negative'

        neg_samples = neg_df.sample(n=len(intersection_df), random_state=42).copy()

        # Combine all DataFrames into one
        combined_data.append(intersection_df)
        combined_data.append(tf1_only_df)
        combined_data.append(tf2_only_df)
        combined_data.append(neg_samples)

# Concatenate all data
combined_df = pd.concat(combined_data)

# Save combined df
output_directory = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/{tf1}_{tf2}/output"
os.makedirs(output_directory, exist_ok=True)
combined_df.to_csv(f"{output_directory}/unfiltered_all_data.csv", index=False)

# Define the path to the proportions.csv file
proportions_file = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/proportions_new.csv"