import pandas as pd
import os
import numpy as np
import subprocess
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Process TF binding data.")
parser.add_argument("tf", type=str, help="Transcription Factor")
parser.add_argument("cell_line", type=str, help="Cell Line")

# Parse arguments
args = parser.parse_args()
tf = args.tf
cell_line_input = args.cell_line

# Directory paths
input_dir = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{tf}/output"
output_dir = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{tf}/tobias/{cell_line_input}"

# Cell line matching
cell_lines = {
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
    "U2OS": "U2OS"
}

# Validate the cell line input
cell_line = None
for key, value in cell_lines.items():
    if value == cell_line_input:
        cell_line = value
        break

if not cell_line:
    raise ValueError(f"Cell line {cell_line_input} is not recognized.")

# Function to filter DataFrame based on given criteria
def filter_df(df, threshold=4000):
    df = df[df['end'] - df['start'] <= threshold]

    if df['count'].max() > 50:
        return df.query('count >= @df["count"].quantile(.75)')
    elif df['count'].max() <= 2:
        return df
    else:
        return df.query('count > @df["count"].median()')

# Function to filter DataFrame based on chromosome
def filter_chromosomes(df, threshold=4000):
    df = df[df['end'] - df['start'] <= threshold]
    valid_chromosomes = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY']
    return df[df['chr'].isin(valid_chromosomes)]

# Initialize an empty list to store the filtered DataFrames
filtered_dfs = []

# Process each .bed file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".bed"):
        # Construct file path
        input_filepath = os.path.join(input_dir, filename)

        # Read the input file
        df = pd.read_csv(input_filepath, sep="\t", header=None, names=['chr', 'start', 'end', 'count'])

        # Filter the DataFrame based on criteria and chromosomes
        filtered_df = filter_df(df)
        filtered_df = filter_chromosomes(filtered_df).copy()

        # Extract cell line name from the file name
        cell_line_key = filename.split('_')[0]

        if cell_lines.get(cell_line_key) == cell_line:
            filtered_df['cell_line'] = cell_line
            filtered_df['label'] = 1

            # Path to the corresponding negative samples file
            negative_filepath = f"/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/{cell_line}/peaks/{cell_line}.filtered.broadPeak"

            if os.path.exists(negative_filepath):
                # Read the negative samples file
                neg_df = pd.read_csv(negative_filepath, sep="\t", usecols=[0, 1, 2], header=None, names=['chr', 'start', 'end'])
                
                # Save the filtered positive dataframe to a temporary file
                temp_filtered_df_file = f'temp_filtered_df_{tf}.bed'
                filtered_df[['chr', 'start', 'end']].to_csv(temp_filtered_df_file, sep='\t', index=False, header=False)

                # Save the negative samples dataframe to a temporary file
                temp_neg_df_file = f'temp_neg_df_{tf}.bed'
                neg_df.to_csv(temp_neg_df_file, sep='\t', index=False, header=False)

                # Use bedtools subtract to find the negative regions
                temp_subtract_output = f'temp_subtract_output_{tf}.bed'
                cmd = ['bedtools', 'subtract', '-a', temp_neg_df_file, '-b', temp_filtered_df_file]
                with open(temp_subtract_output, 'w') as out_f:
                    result = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE)
                    if result.returncode != 0:
                        print(f"Error running bedtools subtract: {result.stderr.decode('utf-8')}")
                        continue

                # Check if the output file was created
                if not os.path.exists(temp_subtract_output):
                    print(f"Output file {temp_subtract_output} not found.")
                    continue

                # Read the output from bedtools subtract
                neg_df = pd.read_csv(temp_subtract_output, sep="\t", header=None, names=['chr', 'start', 'end'])

                # Filter negative samples based on chromosomes
                neg_df = filter_chromosomes(neg_df).copy()

                # Ensure there are enough negative samples
                if len(neg_df) < len(filtered_df):
                    print(f"Not enough negative samples for {cell_line}. Needed: {len(filtered_df)}, Available: {len(neg_df)}. Skipping...")
                    continue

                # Add a dummy 'count' column for consistency
                neg_df['count'] = 1

                # Sample an equal number of negative samples
                neg_samples = neg_df.sample(n=len(filtered_df), random_state=42).copy()
                neg_samples['cell_line'] = cell_line
                neg_samples['label'] = 0

                # Append the negative samples to the filtered DataFrame
                filtered_dfs.append(filtered_df)
                filtered_dfs.append(neg_samples)

                print(f"{filename}: {len(filtered_df)} positive samples, {len(neg_samples)} negative samples")
            else:
                print(f"Negative samples file not found for cell line {cell_line} at {negative_filepath}")

# Concatenate all filtered DataFrames into one
combined_filtered_df = pd.concat(filtered_dfs)

# Reorder columns as specified
combined_filtered_df = combined_filtered_df[['chr', 'start', 'end', 'cell_line', 'label', 'count']]

# Assert that all intervals are <= 4000
assert (combined_filtered_df['end'] - combined_filtered_df['start'] <= 4000).all(), "Some intervals are greater than 4000 base pairs!"

# Save the combined filtered DataFrame to the output file
combined_filtered_df.to_csv(f"{output_dir}/balanced.csv", sep='\t', index=False)

combined_filtered_df.to_csv(f"{output_dir}/balanced.bed", sep='\t', index=False, header=False)

print(f"Combined filtered data saved to {output_dir}")

# Clean up temporary files
os.remove(temp_filtered_df_file)
os.remove(temp_neg_df_file)
os.remove(temp_subtract_output)