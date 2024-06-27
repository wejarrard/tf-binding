import pandas as pd
import os
import numpy as np
import argparse
import subprocess

# Set up argument parser
parser = argparse.ArgumentParser(description="Process TF binding data.")
parser.add_argument("tf", type=str, help="Transcription Factor")
parser.add_argument("--cell_line", type=str, nargs='*', help="Optional cell lines for validation set")

# Parse arguments
args = parser.parse_args()
tf = args.tf
validation_cell_lines = args.cell_line if args.cell_line else []

# Directory paths
input_dir = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{tf}/output"
output_file = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{tf}/entire_set/output_{tf}.csv"

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
        cell_line = cell_lines.get(cell_line_key)

        if cell_line:
            filtered_df['cell_line'] = cell_line
            filtered_df['label'] = 1

            # Path to the corresponding negative samples file
            negative_filepath = f"/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/{cell_line}/peaks/{cell_line}.filtered.broadPeak"

            if os.path.exists(negative_filepath):
                # Read the negative samples file
                neg_df = pd.read_csv(negative_filepath, sep="\t", usecols=[0, 1, 2], header=None, names=['chr', 'start', 'end'])

                # Convert neg_df to a temporary bed file
                temp_bed_file = "./temp.bed"
                neg_df.to_csv(temp_bed_file, sep="\t", index=False, header=False)

                # Construct the command to subtract the bed files
                command = f"bedtools subtract -a {temp_bed_file} -b {input_filepath} > {temp_bed_file}.subtracted"

                # Execute the command using subprocess
                subprocess.run(command, shell=True, check=True)

                # Read the subtracted bed file into a DataFrame
                neg_df = pd.read_csv(f"{temp_bed_file}.subtracted", sep="\t", header=None, names=['chr', 'start', 'end'])

                # Remove the temporary bed files
                os.remove(temp_bed_file)
                os.remove(f"{temp_bed_file}.subtracted")

                # Filter negative samples based on chromosomes
                neg_df = filter_chromosomes(neg_df).copy()

                # Add a dummy 'count' column for consistency
                neg_df['count'] = 1
                neg_df['cell_line'] = cell_line
                neg_df['label'] = 0

                # Append the negative samples to the filtered DataFrame
                filtered_dfs.append(filtered_df)
                filtered_dfs.append(neg_df)

                print(f"{filename}: {len(filtered_df)} positive samples, {len(neg_df)} negative samples")
            else:
                print(f"Negative samples file not found for cell line {cell_line} at {negative_filepath}")

# Concatenate all filtered DataFrames into one
combined_filtered_df = pd.concat(filtered_dfs)

# Reorder columns as specified
combined_filtered_df = combined_filtered_df[['chr', 'start', 'end', 'cell_line', 'label', 'count']]

# Assert that all intervals are <= 4000
assert (combined_filtered_df['end'] - combined_filtered_df['start'] <= 4000).all(), "Some intervals are greater than 4000 base pairs!"

# Save the combined filtered DataFrame to the output file
os.makedirs(f'/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{tf}/entire_set', exist_ok=True)
combined_filtered_df.to_csv(output_file, sep='\t', index=False)

print(f"Combined filtered data saved to {output_file}")

# Generate training and validation sets
if validation_cell_lines:
    # Combine all specified cell lines into a single validation set
    validation_set = combined_filtered_df[combined_filtered_df['cell_line'].isin(validation_cell_lines)]
    training_set = combined_filtered_df[~combined_filtered_df['cell_line'].isin(validation_cell_lines)]
    validation_set_file = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{tf}/entire_set/validation_combined.csv"
    training_set_file = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{tf}/entire_set/training_combined.csv"
    validation_set.to_csv(validation_set_file, sep='\t', index=False)
    training_set.to_csv(training_set_file, sep='\t', index=False)
    print(f"Combined validation set and training set generated with specified cell lines: {validation_cell_lines}")
else:
    for cell_line in cell_lines.values():
        cell_line_data = combined_filtered_df[combined_filtered_df['cell_line'] == cell_line]

        if len(cell_line_data) == 0:
            print(f"{cell_line}: did not make threshold, skipping")
            continue

        # Validation set is the current cell line
        validation_set = cell_line_data
        # Training set is all other cell lines
        training_set = combined_filtered_df[combined_filtered_df['cell_line'] != cell_line]
        # Save the training and validation sets
        training_set_file = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{tf}/entire_set/training_{cell_line}.csv"
        validation_set_file = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{tf}/entire_set/validation_{cell_line}.csv"
        training_set.to_csv(training_set_file, sep='\t', index=False)
        validation_set.to_csv(validation_set_file, sep='\t', index=False)

        print(f"{cell_line}: Training and validation sets generated")
