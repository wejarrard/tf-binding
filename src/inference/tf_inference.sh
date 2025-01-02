#!/bin/bash

# Set strict error handling
set -euo pipefail

# Function to handle cleanup
cleanup() {
    echo "Cleaning up..."
    # GNU Parallel will handle child process cleanup
    exit 1
}

# Trap signals
trap cleanup SIGINT SIGTERM

# Function to check and setup conda environment
setup_conda() {
    if ! command -v conda &> /dev/null; then
        echo "Error: conda not found. Please install conda first."
        exit 1
    fi

    if conda env list | grep -q "processing"; then
        echo "Activating processing environment..."
        source activate processing
    else
        echo "Creating processing environment..."
        conda env create -f "${path_to_project}/environment.yml"
        conda activate processing
    fi
}

# Function to get input bed file path
get_input_bed_file() {
    local identifier="$1"
    
    if [[ "$identifier" =~ ^SRR ]]; then
        echo "/data1/datasets_1/human_prostate_PDX/processed/ATAC/${identifier}/peaks/${identifier}.filtered.broadPeak"
    else
        echo "/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/${identifier}/peaks/${identifier}.filtered.broadPeak"
    fi
}

# Function to process a single model-cell line combination
process_model_cell_line() {
    local CELL_LINE="$1"
    local MODEL="$2"
    local path_to_project="$3"
    local no_ground_truth="$4"
    local balance="$5"
    local positive_only="$6"
    local rewrite="$7"
    local MODEL_JSON_PATH="$8"
    
    echo "Processing: $CELL_LINE - $MODEL"
    
    # Error handling
    set -e
    
    # Get input bed file path
    input_bed_file=$(get_input_bed_file "$CELL_LINE")
    if [ ! -f "${input_bed_file}" ]; then
        echo "Error: Input bed file ${input_bed_file} does not exist"
        return 1
    fi

    # Extract TF name
    TF_NAME=$(echo "$MODEL" | cut -d'-' -f1)
    echo "Processing TF: $TF_NAME"
    
    # Set input file name based on conditions
    if [ "$balance" = TRUE ]; then
        input_file="${TF_NAME}-${CELL_LINE}-BALANCED"
    else
        if [ "$positive_only" = TRUE ]; then
            input_file="${TF_NAME}-${CELL_LINE}-POSITIVE-ONLY"
        else
            input_file="${TF_NAME}_${CELL_LINE}"
        fi
    fi
    
    # Check if processing can be skipped
    if [ "$rewrite" = FALSE ] && [ -d "${path_to_project}/data/jsonl/${input_file}" ]; then
        echo "Output exists for ${input_file}, skipping preprocessing..."
    else
        # Get model path from JSON
        MODEL_PATH=$(jq -r --arg model "$MODEL" '.[$model]' "$MODEL_JSON_PATH")
        if [ -z "${MODEL_PATH}" ] || [ "${MODEL_PATH}" = "null" ]; then
            echo "Model path for $MODEL not found in $MODEL_JSON_PATH"
            return 1
        fi

        # Create necessary directories
        mkdir -p "${path_to_project}/data/data_splits"
        mkdir -p "${path_to_project}/data/jsonl/${input_file}"
        
        echo "Processing data for ${input_file}..."
        
        # Generate training peaks based on conditions
        if [ "$no_ground_truth" = TRUE ]; then
            python "${path_to_project}/src/processing/generate_training_peaks.py" \
                --no_ground_truth \
                --input_bed_file "${input_bed_file}" \
                --output_dir "${path_to_project}/data/data_splits" \
                --output_file "${input_file}.csv"
        else
            cmd="python ${path_to_project}/src/processing/generate_training_peaks.py \
                --tf ${TF_NAME} \
                --validation_cell_lines ${CELL_LINE} \
                --validation_file ${input_file}.csv"
            
            [ "$balance" = TRUE ] && cmd+=" --balance"
            [ "$positive_only" = TRUE ] && cmd+=" --positive_only"
            
            eval "$cmd"
        fi
        
        # Move and process files
        mv -f "${path_to_project}/data/data_splits/${input_file}.csv" \
            "${path_to_project}/data/data_splits/${input_file}_no_motifs.csv"
        
        # Add motifs
        python "${path_to_project}/src/inference/motif_finding/motif_addition.py" \
            --tsv_file "${path_to_project}/data/data_splits/${input_file}_no_motifs.csv" \
            --jaspar_file "${path_to_project}/src/inference/motif_finding/motif.jaspar" \
            --reference_genome "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/genome.fa" \
            --output_file "${path_to_project}/data/data_splits/${input_file}.csv" \
            --min_score -10
        
        # Prepare data command
        prepare_data_cmd="python ${path_to_project}/src/inference/prepare_data.py \
            --input_file ${input_file}.csv \
            --output_path ${path_to_project}/data/jsonl/${input_file}"
        
        if [[ "$CELL_LINE" =~ ^SRR ]]; then
            prepare_data_cmd+=" --cell_line_dir /data1/datasets_1/human_prostate_PDX/processed/ATAC"
        else
            prepare_data_cmd+=" --cell_line_dir /data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines"
        fi
        
        eval "$prepare_data_cmd"
        
        # Set permissions
        chgrp users "${path_to_project}/data/jsonl/${input_file}"
    fi
    
    echo "Running inference for ${input_file}..."
    echo "Using model path: ${MODEL_PATH}"
    
    # Run inference and capture job name
    job_name=$(python "${path_to_project}/src/inference/aws_inference.py" \
        --model "${MODEL}" \
        --sample "${CELL_LINE}" \
        --model_paths_file "${MODEL_JSON_PATH}" \
        --project_path "${path_to_project}" \
        --local_dir "${path_to_project}/data/jsonl/${input_file}")
        
    echo "Completed: $CELL_LINE - $MODEL"
}

# Main function to orchestrate the entire process
main() {
    # Configuration
    path_to_project="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"
    MODEL_JSON_PATH="${path_to_project}/src/inference/models.json"
    
    # Cell lines and models
    CELL_LINES=("SRR12455433" "SRR12455434" "SRR12455435" "SRR12455432" "SRR12455437")
                # "SRR12455436" "SRR12455439" "SRR12455440" "SRR12455441" "SRR12455442" "SRR12455445")
    MODELS_TO_USE=("NEUROD1-chr3")
    
    # Processing flags
    no_ground_truth=TRUE
    balance=FALSE
    positive_only=FALSE
    rewrite=FALSE
    
    # Setup conda environment
    setup_conda
    
    # Create download queue file
    rm -f "${path_to_project}/data/download_queue.txt"
    touch "${path_to_project}/data/download_queue.txt"
    
    # Export functions and variables for parallel
    export -f process_model_cell_line
    export -f get_input_bed_file
    export path_to_project no_ground_truth balance positive_only rewrite MODEL_JSON_PATH
    
    # Create combinations file
    temp_file=$(mktemp)
    for cell_line in "${CELL_LINES[@]}"; do
        for model in "${MODELS_TO_USE[@]}"; do
            echo "$cell_line $model $path_to_project $no_ground_truth $balance $positive_only $rewrite $MODEL_JSON_PATH"
        done
    done > "$temp_file"
    
    # Run processing with parallel
    echo "Starting parallel processing..."
    parallel --progress --jobs 4 --colsep ' ' \
        process_model_cell_line {1} {2} {3} {4} {5} {6} {7} {8} < "$temp_file"
    
    # Cleanup
    rm -f "$temp_file"
    rm -f "${path_to_project}/data/download_queue.txt"
    
    echo "All processing complete."
}

# Run main function
main "$@"