#!/bin/bash

# Set up signal handling
cleanup() {
    echo "Cleaning up..."
    # Kill all child processes
    pkill -P $$
    # Remove PID file
    rm -f "./model_processing.pid"
    exit 1
}

# Catch termination signals
trap cleanup SIGINT SIGTERM

# Save main process PID
echo $$ > "./model_processing.pid"
echo "Process ID saved to ./model_processing.pid"
echo "To kill this process and all children, run: kill -TERM \$(cat ./model_processing.pid)"

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
CELL_LINES=("SRR12455433" "SRR12455434" "SRR12455435" "SRR12455432" "SRR12455437" "SRR12455436" "SRR12455439" "SRR12455440" "SRR12455441" "SRR12455442" "SRR12455445")

# CELL_LINES=("VCAP")
# Models to use
MODELS_TO_USE=("NEUROD1-chr3")

no_ground_truth=TRUE

# Only relevant if no_ground_truth is FALSE (for cell lines with ground truth (chip seq))
balance=FALSE
positive_only=FALSE
path_to_project="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"

# Add rewrite option with default FALSE
rewrite=FALSE

################################################################################
########################### Variable Setup #####################################
################################################################################

# Path to model.json
MODEL_JSON_PATH="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/inference/models.json"

# Function to determine input bed file path
get_input_bed_file() {
    local identifier="$1"
    local input_bed_file=""
    
    # Check if identifier starts with "SRR" (experiment)
    if [[ "$identifier" =~ ^SRR ]]; then
        input_bed_file="/data1/datasets_1/human_prostate_PDX/processed/ATAC/${identifier}/peaks/${identifier}.filtered.broadPeak"
    else
        # Assume it's a cell line
        input_bed_file="/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/${identifier}/peaks/${identifier}.filtered.broadPeak"
    fi
    
    echo "$input_bed_file"
}

# Modified process_model_cell_line function with error handling
process_model_cell_line() {
    local CELL_LINE="$1"
    local MODEL="$2"
    
    echo "Starting process for $CELL_LINE - $MODEL (PID: $$)"
    
    # Add error handling
    set -e

    echo "Processing cell line: $CELL_LINE with model: $MODEL"

    # Get correct input bed file path
    input_bed_file=$(get_input_bed_file "$CELL_LINE")
    echo "Using input bed file: ${input_bed_file}"

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

    # Check if final output already exists and skip if rewrite=FALSE
    if [ "$rewrite" = FALSE ] && [ -d "${path_to_project}/data/jsonl/${input_file}" ]; then
        echo "Output directory already exists for ${input_file}. Skipping processing..."
        return 0
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

    # Prepare data based on input type
    prepare_data_cmd="python ${path_to_project}/src/inference/prepare_data.py --input_file ${input_file}.csv --output_path ${path_to_project}/data/jsonl/${input_file}"

    # Set cell_line_dir based on input type
    if [[ "$CELL_LINE" =~ ^SRR ]]; then
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

    echo "Finished process for $CELL_LINE - $MODEL (PID: $$)"
}

# Modified job management with better tracking
MAX_JOBS=4  # Adjust this number as needed
declare -A active_pids

wait_for_jobs() {
    while [ ${#active_pids[@]} -ge $MAX_JOBS ]; do
        for pid in "${!active_pids[@]}"; do
            if ! kill -0 $pid 2>/dev/null; then
                unset active_pids[$pid]
            fi
        done
        sleep 1
    done
}

# Main processing loop with improved job tracking
for CELL_LINE in "${CELL_LINES[@]}"; do
    for MODEL in "${MODELS_TO_USE[@]}"; do
        wait_for_jobs
        process_model_cell_line "$CELL_LINE" "$MODEL" &
        pid=$!
        active_pids[$pid]="$CELL_LINE-$MODEL"
        echo "Started job for $CELL_LINE-$MODEL with PID: $pid"
    done
done

# Wait for remaining jobs
wait

echo "Starting downloads from S3..."

# Create directory for processed results if it doesn't exist
mkdir -p "${path_to_project}/data/processed_results"

# Loop through all combinations and download results
for CELL_LINE in "${CELL_LINES[@]}"; do
    for MODEL in "${MODELS_TO_USE[@]}"; do
        echo "Downloading results for ${MODEL}-${CELL_LINE}..."
        # Now we only pass the model and cell line - job name will be read from JSON file
        python download_results.py "$MODEL" "$CELL_LINE"
    done
done

# Cleanup at the end
rm -f "./model_processing.pid"

echo "All processing complete."