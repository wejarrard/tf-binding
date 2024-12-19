#!/bin/bash

# Base directory where the cell line directories are located
base_dir="/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines" # Replace with the actual base directory path

# Path to the JSON mapping file
json_file="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/processing/cell_line_mapping.json"

# Check if jq is installed
if ! command -v jq &> /dev/null
then
    echo "Error: jq could not be found. Please install jq to proceed."
    exit 1
fi

# Read the cell lines from the JSON mapping file
CELL_LINES=($(jq -r '.[]' "$json_file"))

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
    src_path="${base_dir}/${cell_line}/pileup_mod/"

    # Destination path on S3
    dest_path="s3://tf-binding-sites/pretraining/data/cell_lines/${cell_line}/pileup"

    echo "Syncing files for ${cell_line}..."
    aws s3 sync "$src_path" "$dest_path" --exclude "*" --include "*.gz" --include "*.tbi" --size-only
done

echo "Deletion and sync completed for all cell lines."