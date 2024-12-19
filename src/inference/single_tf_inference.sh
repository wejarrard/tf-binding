#!/bin/bash

# Activate conda environment if needed
if conda env list | grep -q "processing"; then
    source activate processing
else
    conda env create -f "${path_to_project}/environment.yml"
    conda activate processing
fi


################################################################################
############################## Inputs ##########################################
################################################################################

# NEURO CELL LINES
# CELL_LINES=("SRR12455433" "SRR12455434" "SRR12455435" "SRR12455432" "SRR12455437")
# ADENO CELL LINES
# CELL_LINES=("SRR12455436" "SRR12455439" "SRR12455440" "SRR12455441" "SRR12455442" "SRR12455445")
# BOTH
# CELL_LINES=("SRR12455433" "SRR12455434" "SRR12455435" "SRR12455432" "SRR12455437" "SRR12455436" "SRR12455439" "SRR12455440" "SRR12455441" "SRR12455442" "SRR12455445")

CELL_LINES=("VCAP")
# Models to use
MODELS_TO_USE=("RB1")

no_ground_truth=FALSE

# Only relevant if no_ground_truth is FALSE (for cell lines with ground truth (chip seq))
balance=FALSE
positive_only=FALSE
path_to_project="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"

################################################################################
########################### Variable Setup #####################################
################################################################################

# Path to model.json
MODEL_JSON_PATH="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/inference/models.json"

# Function to process a single combination of CELL_LINE and MODEL
process_model_cell_line() {
    local CELL_LINE="$1"
    local MODEL="$2"

    echo "Processing cell line: $CELL_LINE with model: $MODEL"

    # First check if input bed file exists
    input_bed_file="/data1/datasets_1/human_prostate_PDX/processed/ATAC/${CELL_LINE}/peaks/${CELL_LINE}.filtered.broadPeak"
    input_bed_file="/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/${CELL_LINE}/peaks/${CELL_LINE}.filtered.broadPeak"
    if [ ! -f "${input_bed_file}" ]; then
        echo "Error: Input bed file ${input_bed_file} does not exist"
        return 1
    fi

    # Extract TF_NAME as everything before first dash
    TF_NAME=$(echo "$MODEL" | cut -d'-' -f1)
    echo "Extracted TF_NAME: $TF_NAME from MODEL: $MODEL"

    # Set input_file based on conditions
    if [ "$balance" = TRUE ]; then
        input_file="${TF_NAME}-${CELL_LINE}-BALANCED"
    else
        if [ "$positive_only" = TRUE ]; then
            input_file="${TF_NAME}-${CELL_LINE}-POSITIVE-ONLY"
        else
            input_file="${TF_NAME}_${CELL_LINE}"
        fi
    fi

    # Get the model path from model.json
    MODEL_PATH=$(jq -r --arg model "$MODEL" '.[$model]' "$MODEL_JSON_PATH")

    # Check if MODEL_PATH is empty or null
    if [ -z "${MODEL_PATH}" ] || [ "${MODEL_PATH}" = "null" ]; then
        echo "Model path for $MODEL not found in $MODEL_JSON_PATH"
        return 1
    fi

    # Create the job name using MODEL and CELL_LINE
    JOB_NAME="${MODEL}-${CELL_LINE}"

    # Build MODEL_PATHS_JSON for the current model
    MODEL_PATHS_JSON="{\"${JOB_NAME}\": \"${MODEL_PATH}\"}"

    # Create output directories if they don't exist
    mkdir -p "${path_to_project}/data/data_splits"
    mkdir -p "${path_to_project}/data/jsonl/${input_file}"

    echo "Processing data for ${input_file}..."
    # Run the Python script with the appropriate arguments based on conditions
    if [ "$no_ground_truth" = TRUE ]; then
        python "${path_to_project}/src/processing/generate_training_peaks.py" \
        --no_ground_truth \
        --input_bed_file "${input_bed_file}" \
        --output_dir "${path_to_project}/data/data_splits" \
        --output_file "${input_file}.csv"
    else
        # Set base arguments
        cmd="python ${path_to_project}/src/processing/generate_training_peaks.py --tf ${TF_NAME} --validation_cell_lines ${CELL_LINE} --validation_file ${input_file}.csv"

        # Add balance or positive_only options as needed
        [ "$balance" = TRUE ] && cmd+=" --balance"
        [ "$positive_only" = TRUE ] && cmd+=" --positive_only"

        # Run command
        eval "$cmd"
    fi

    # Move and rename the file locally
    mv -f "${path_to_project}/data/data_splits/${input_file}.csv" "${path_to_project}/data/data_splits/${input_file}_no_motifs.csv"

    # Continue running local operations for motif addition
    python "${path_to_project}/src/inference/motif_finding/motif_addition.py" \
    --tsv_file "${path_to_project}/data/data_splits/${input_file}_no_motifs.csv" \
    --jaspar_file "${path_to_project}/src/inference/motif_finding/motif.jaspar" \
    --reference_genome "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/genome.fa" \
    --output_file "${path_to_project}/data/data_splits/${input_file}.csv" \
    --min_score -10

    # Prepare data based on no_ground_truth setting
    prepare_data_cmd="python ${path_to_project}/src/inference/prepare_data.py --input_file ${input_file}.csv --output_path ${path_to_project}/data/jsonl/${input_file}"

    if [ "$no_ground_truth" = TRUE ]; then
        prepare_data_cmd+=" --cell_line_dir /data1/datasets_1/human_prostate_PDX/processed/ATAC"
    else
        prepare_data_cmd+=" --cell_line_dir /data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines"
    fi

    # Run prepare data command
    eval "$prepare_data_cmd"

    chgrp users "${path_to_project}/data/jsonl/${input_file}"

    # Call inference
    python "${path_to_project}/src/inference/aws_inference.py" \
    --model_paths "${MODEL_PATHS_JSON}" \
    --project_path "${path_to_project}" \
    --local_dir "${path_to_project}/data/jsonl/${input_file}"

    echo "Finished processing cell line: $CELL_LINE with model: $MODEL"
}

# Maximum number of parallel jobs
MAX_JOBS=4  # Adjust this number as needed

# Function to wait for jobs to finish when the maximum number is reached
wait_for_jobs() {
    local MAX_JOBS=$1
    while [ $(jobs -p | wc -l) -ge $MAX_JOBS ]; do
        sleep 1
    done
}

# Loop through each cell line and model combination
for CELL_LINE in "${CELL_LINES[@]}"; do
    for MODEL in "${MODELS_TO_USE[@]}"; do
        # Wait for available job slot
        wait_for_jobs $MAX_JOBS
        # Start the process in the background
        process_model_cell_line "$CELL_LINE" "$MODEL" &
    done
done

# Wait for all background jobs to finish
wait

echo "All processing complete."