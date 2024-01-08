#!/bin/bash

# Define the array of cell line directories
CELL_LINES=(
    "NCI-H660"
    "22Rv1"
    "LNCAP"
    "PC-3"
    "C42B"
    "C4-2"
    "MCF7"
    "Ramos"
    "A549"
    "HT-1376"
    "K-562"
    "JURKAT"
    "Hep_G2"
    "MCF_10A"
    "SCaBER"
    "SEM"
    "786-O"
    "Ishikawa"
    "MOLT-4"
    "BJ_hTERT"
    "SIHA"
    "Detroit_562"
    "OVCAR-8"
    "PANC-1"
    "NCI-H69"
    "HELA"
    "HuH-7"
    "THP-1"
    "U-87_MG"
    "SK-N-SH"
    "TC-32"
    "RS411"
    "TTC1240"
)

# Base directory where the cell line directories are located
base_dir="/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines" # Replace with the actual base directory path

# Loop through each cell line
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
