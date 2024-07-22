#!/bin/bash

# Define the array of cell line directories
CELL_LINES=(
    "HEK_293"
    "A549"
    "HuH-7"
    "K562"
    "MCF7"
    "22Rv1"
    "A-375"
    "C4-2"
    "LNCAP"
    "PC-3"
    "THP-1"
    "UM-UC-3"
    "VCAP"
    "CAMA-1"
    "AN3_CA"
    "HCC1954"
    "HEC-1-A"
    "Ishikawa"
    "MDA-MB-134-VI"
    "MDA-MB-453"
    "NCI-H3122"
    "NCI-H441"
    "RT4"
    "SK-BR-3"
    "GP5D"
    "LS-180"
    "CFPAC-1"
    "GM12878"
    "MDA-MB-231"
    "HELA"
    "SK-N-SH"
    "HMEC"
    "U2OS"
)

# Base directory where the cell line directories are located
base_dir="/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines" # Replace with the actual base directory path

# Get a list of all cell lines currently in S3
existing_cell_lines=$(aws s3 ls s3://tf-binding-sites/pretraining/data/cell_lines/ | awk '{print $2}' | sed 's#/##')

# Loop through each existing cell line and remove it if it's not in the CELL_LINES array
for existing_cell_line in $existing_cell_lines; do
    if [[ ! " ${CELL_LINES[@]} " =~ " ${existing_cell_line} " ]]; then
        echo "Removing files for ${existing_cell_line} from S3..."
        aws s3 rm "s3://tf-binding-sites/pretraining/data/cell_lines/${existing_cell_line}/" --recursive
    fi
done

# Loop through each cell line in the list and sync files
for cell_line in "${CELL_LINES[@]}"; do
    # Source directory path
    src_path="${base_dir}/${cell_line}/pileup_mod_log10/"

    # Destination path on S3
    dest_path="s3://tf-binding-sites/pretraining/data/cell_lines/${cell_line}/pileup"

    # Delete .tbi files in the destination S3 bucket
    echo "Deleting .tbi files for ${cell_line}..."
    aws s3 rm "$dest_path" --recursive --exclude "*" --include "*.tbi"
    aws s3 rm "$dest_path" --recursive --exclude "*" --include "*.gz"

    # Sync .gz and .tbi files to S3
    echo "Uploading .gz and .tbi files for ${cell_line}..."
    aws s3 sync "$src_path" "$dest_path" --exclude "*" --include "*.gz"
    aws s3 sync "$src_path" "$dest_path" --exclude "*" --include "*.tbi"
done

echo "Deletion and sync completed for all cell lines."

# aws s3 cp /data1/home/wjarrard/projects/preprocessing/data/combined.bed s3://tf-binding-sites/pretraining/data_basic/combined.bed
