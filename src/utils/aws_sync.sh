#!/bin/bash

# Base directory where the cell line directories are located
base_dir="/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines"

# Path to the JSON mapping file
json_file="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/utils/cell_line_mapping.json"

# Check if jq is installed
if ! command -v jq &> /dev/null
then
    echo "Error: jq could not be found. Please install jq to proceed."
    exit 1
fi

# Read the cell lines from the JSON mapping file
CELL_LINES=($(jq -r '.[]' "$json_file"))

# Loop through each cell line in the list and sync files
for cell_line in "${CELL_LINES[@]}"; do
    # Source directory path
    src_path="${base_dir}/${cell_line}/pileup_mod/"

    # Destination path on S3
    dest_path="s3://tf-binding-sites/pretraining/data/cell_lines/${cell_line}/pileup"

    echo "Syncing files for ${cell_line}..."
    aws s3 sync "$src_path" "$dest_path" --exclude "*" --include "*.gz" --include "*.tbi" --size-only --delete
done

echo "Sync completed for all cell lines."