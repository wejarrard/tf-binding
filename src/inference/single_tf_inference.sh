#!/bin/bash

# Inputs
TF_NAME='AR'
CELL_LINE='22Rv1'
MODEL_PATHS_JSON="{
    \"Motifs-${TF_NAME}-${CELL_LINE}\": \"s3://tf-binding-sites/finetuning/results/output/AR-Full-Data-Model-2024-10-17-04-20-48-548/output/model.tar.gz\"
}"
balance=FALSE
positive_only=FALSE
path_to_project="/Users/wejarrard/projects/tf-binding"

# Copy the script to the remote server
scp -r "${path_to_project}/src/processing/generate_training_peaks.py" ucsf:/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/generate_training_peaks.py

# Execute commands on the remote machine via SSH and pass variables
ssh ucsf <<ENDSSH
    # Variables are passed from the local environment
    TF_NAME="${TF_NAME}"
    CELL_LINE="${CELL_LINE}"
    balance="${balance}"
    positive_only="${positive_only}"

    # Navigate to the directory with the Python script
    cd /data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/
    
    # Activate conda environment
    source activate processing
    
    # Run the Python script with the appropriate arguments
    if [ "\$balance" = TRUE ]; then
        python generate_training_peaks.py "\${TF_NAME}" --balance --validation_cell_lines "\${CELL_LINE}"
    else
    if [ "\$positive_only" = TRUE ]; then
        python generate_training_peaks.py "\${TF_NAME}" --positive_only --validation_cell_lines "\${CELL_LINE}"
    else
        python generate_training_peaks.py "\${TF_NAME}" --validation_cell_lines "\${CELL_LINE}"
    fi
    fi
ENDSSH

# Switch back to running code on the local machine
echo "Remote process completed, now running local operations..."

# Example: Download the validation_combined.csv file via SCP after the remote execution
scp ucsf:/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/${TF_NAME}/entire_set/validation_combined.csv "${path_to_project}/data/data_splits"

# Move and rename the file locally
mv "${path_to_project}/data/data_splits/validation_combined.csv" "${path_to_project}/data/data_splits/validation_combined_no_motifs.csv"

# Continue running local operations
# Run a local Python script for motif addition
python "${path_to_project}/src/inference/motif_finding/motif_addition.py"

# PASS IN JSON HERE
# Pass the MODEL_PATHS_JSON into the inference.py script
python "${path_to_project}/src/inference/inference.py" --model_paths "${MODEL_PATHS_JSON}"