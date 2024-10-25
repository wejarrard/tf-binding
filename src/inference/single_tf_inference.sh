#!/bin/bash

# Inputs
TF_NAME='AR'
CELL_LINE='VCAP'
MODEL_PATHS_JSON="{
    \"Motifs-${TF_NAME}-${CELL_LINE}\": \"s3://tf-binding-sites/finetuning/results/output/AR-Full-Data-Model-2024-10-17-04-20-48-548/output/model.tar.gz\"
}"
balance=FALSE
positive_only=FALSE
path_to_project="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"

# Activate conda environment
# if conda environment named processing exists, activate it, else install it at path_to_project/environment.yml
if conda env list | grep -q "processing"; then
    conda activate processing
else
    conda env create -f "${path_to_project}/environment.yml"
    conda activate processing
fi

# Run the Python script with the appropriate arguments
if [ "$balance" = TRUE ]; then
    python "${path_to_project}/src/processing/generate_training_peaks.py" "${TF_NAME}" --balance --validation_cell_lines "${CELL_LINE}"
else
    if [ "$positive_only" = TRUE ]; then
        python "${path_to_project}/src/processing/generate_training_peaks.py" "${TF_NAME}" --positive_only --validation_cell_lines "${CELL_LINE}"
    else
        python "${path_to_project}/src/processing/generate_training_peaks.py" "${TF_NAME}" --validation_cell_lines "${CELL_LINE}"
    fi
fi

# Move and rename the file locally
mv -f "${path_to_project}/data/data_splits/validation_combined.csv" "${path_to_project}/data/data_splits/validation_combined_no_motifs.csv"

# Continue running local operations
# Run a local Python script for motif addition
python "${path_to_project}/src/inference/motif_finding/motif_addition.py" \
--tsv_file "${path_to_project}/data/data_splits/validation_combined_no_motifs.csv" \
--jaspar_file "${path_to_project}/src/inference/motif_finding/motif.jaspar" \
--reference_genome "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/genome.fa" \
--output_file "${path_to_project}/data/data_splits/validation_combined.csv" \
--min_score -10

python "${path_to_project}/src/inference/prepare_data.py" \
--input_file "validation_combined.csv"

# Call inference
python "${path_to_project}/src/inference/aws_inference.py" \
--model_paths "${MODEL_PATHS_JSON}" \
--project_path "${path_to_project}" \
--local_dir "${path_to_project}/data/jsonl"
