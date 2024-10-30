#!/bin/bash

# Inputs
TF_NAME='AR'
CELL_LINE='SRR12455437'
input_file="AR_SRR12455437"
MODEL_PATHS_JSON="{
    \"Motifs-${TF_NAME}-${CELL_LINE}\": \"s3://tf-binding-sites/finetuning/results/output/AR-Full-Data-Model-2024-10-17-04-20-48-548/output/model.tar.gz\"
}"


no_ground_truth=TRUE
input_bed_file="/data1/datasets_1/human_prostate_PDX/processed/ATAC/${CELL_LINE}/peaks/${CELL_LINE}.filtered.broadPeak"

# Only relevent if no_ground_truth is FALSE
balance=TRUE
positive_only=FALSE
path_to_project="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"


# Activate conda environment
if conda env list | grep -q "processing"; then
    conda activate processing
else
    conda env create -f "${path_to_project}/environment.yml"
    conda activate processing
fi

# Run the Python script with the appropriate arguments
if [ "$no_ground_truth" = TRUE ]; then
    python "${path_to_project}/src/processing/generate_training_peaks.py" \
    --no_ground_truth \
    --input_bed_file "${input_bed_file}" \
    --output_dir "${path_to_project}/data/data_splits" \
    --output_file "${input_file}.csv"
else
    if [ "$balance" = TRUE ]; then
        python "${path_to_project}/src/processing/generate_training_peaks.py" --tf "${TF_NAME}" --balance --validation_cell_lines "${CELL_LINE}" --validation_file "${input_file}.csv"
    else
        if [ "$positive_only" = TRUE ]; then
            python "${path_to_project}/src/processing/generate_training_peaks.py" --tf "${TF_NAME}" --positive_only --validation_cell_lines "${CELL_LINE}" --validation_file "${input_file}.csv"
        else
            python "${path_to_project}/src/processing/generate_training_peaks.py" --tf "${TF_NAME}" --validation_cell_lines "${CELL_LINE}" --validation_file "${input_file}.csv"
        fi
    fi
fi

# Move and rename the file locally
mv -f "${path_to_project}/data/data_splits/${input_file}.csv" "${path_to_project}/data/data_splits/${input_file}_no_motifs.csv"

# Continue running local operations
# Run a local Python script for motif addition
python "${path_to_project}/src/inference/motif_finding/motif_addition.py" \
--tsv_file "${path_to_project}/data/data_splits/${input_file}_no_motifs.csv" \
--jaspar_file "${path_to_project}/src/inference/motif_finding/motif.jaspar" \
--reference_genome "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/genome.fa" \
--output_file "${path_to_project}/data/data_splits/${input_file}.csv" \
--min_score -10


if [ "$no_ground_truth" = TRUE ]; then
    python "${path_to_project}/src/inference/prepare_data.py" \
    --input_file "${input_file}.csv" \
    --output_path "${path_to_project}/data/jsonl/${input_file}" \
    --cell_line_dir "/data1/datasets_1/human_prostate_PDX/processed/ATAC"
else
    python "${path_to_project}/src/inference/prepare_data.py" \
    --input_file "${input_file}.csv" \
    --output_path "${path_to_project}/data/jsonl/${input_file}" \
    --cell_line_dir "/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines"
fi


# Call inference
python "${path_to_project}/src/inference/aws_inference.py" \
--model_paths "${MODEL_PATHS_JSON}" \
--project_path "${path_to_project}" \
--local_dir "${path_to_project}/data/jsonl/${input_file}"