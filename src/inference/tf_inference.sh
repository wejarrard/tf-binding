#!/bin/bash

set -euo pipefail

# Configuration
PROJECT_PATH="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"
MODEL_JSON_PATH="${PROJECT_PATH}/src/inference/models.json"

# Process a single combination of model and cell line
process_combination() {
    local cell_line="$1"
    local model="$2"
    
    # Extract SRR value if present
    local srr_value=""
    if [[ "$cell_line" =~ : ]]; then
        srr_value=$(echo "$cell_line" | cut -d':' -f2)
        cell_line=$(echo "$cell_line" | cut -d':' -f1)
    fi
    
    # Setup logging
    local log_dir="${PROJECT_PATH}/logs/inference"
    local log_file="${log_dir}/${cell_line}_${model}.log"
    mkdir -p "$log_dir"

    # remove log file if it exists
    if [ -f "$log_file" ]; then
        rm "$log_file"
    fi
    
    # Log start of processing
    {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting processing: ${cell_line} - ${model}"
        echo "SRR_VALUE: ${srr_value}"
        echo "Processing TF: ${model%%-*}"
    } >> "$log_file"
    
    # Get input bed file path
    local bed_path
    if [[ "$cell_line" =~ ^LuCaP ]]; then
        bed_path="/data1/datasets_1/human_prostate_PDX/processed/ATAC/${cell_line}/${srr_value}/peaks/${srr_value}.filtered.broadPeak"
    else
        bed_path="/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/${cell_line}/peaks/${cell_line}.filtered.broadPeak"
    fi
    
    # Validate input file
    if [ ! -f "${bed_path}" ]; then
        echo "Error: Input bed file ${bed_path} not found" | tee -a "$log_file"
        return 1
    fi
    
    # Extract TF name and set input file name
    local tf_name=$(echo "$model" | cut -d'-' -f1)
    local input_file="${model}_${cell_line}"
    
    # Skip if output exists and rewrite is false
    if [ "${REWRITE:-false}" = false ] && [ -d "${PROJECT_PATH}/data/jsonl/${input_file}" ]; then
        echo "Output exists for ${input_file}, skipping preprocessing..." | tee -a "$log_file"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running inference for ${input_file}..." >> "$log_file"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using model path: " >> "$log_file"
    fi
    
    # Prepare directories
    mkdir -p "${PROJECT_PATH}/data/data_splits" "${PROJECT_PATH}/data/jsonl/${input_file}"
    
    # Generate training peaks
    python "${PROJECT_PATH}/src/utils/generate_training_peaks.py" \
        --no_ground_truth \
        --input_bed_file "${bed_path}" \
        --output_dir "${PROJECT_PATH}/data/data_splits" \
        --output_file "${input_file}.csv" 2>&1 | tee -a "$log_file"
    
    # Prepare data
    local cell_line_dir
    if [[ "$cell_line" =~ ^LuCaP ]]; then
        cell_line_dir="/data1/datasets_1/human_prostate_PDX/processed/ATAC/${cell_line}"
    else
        cell_line_dir="/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines"
    fi
    
    python "${PROJECT_PATH}/src/inference/prepare_data.py" \
        --input_file "${input_file}.csv" \
        --output_path "${PROJECT_PATH}/data/jsonl/${input_file}" \
        --cell_line_dir "${cell_line_dir}" 2>&1 | tee -a "$log_file"
    
    # Run inference
    python "${PROJECT_PATH}/src/inference/aws_inference.py" \
        --model "${model}" \
        --sample "${cell_line}" \
        --model_paths_file "${MODEL_JSON_PATH}" \
        --project_path "${PROJECT_PATH}" \
        --local_dir "${PROJECT_PATH}/data/jsonl/${input_file}" 2>&1 | tee -a "$log_file"
}

main() {
    # Activate conda environment
    if ! command -v conda &> /dev/null; then
        echo "Error: conda not found"
        exit 1
    fi
    
    source activate processing 2>/dev/null || conda env create -f "${PROJECT_PATH}/environment.yml"
    
    # Define cell lines and models
    # local cell_lines=("LuCaP_81:SRR12455442") #"LuCaP_78:SRR12455441" "LuCaP_77CR:SRR12455440")
    local cell_lines=("LuCaP_81:SRR12455442" "LuCaP_78:SRR12455441" "LuCaP_77CR:SRR12455440")
    # local cell_lines=("22Rv1")
    local models=("AR-log10-new")
    
    # Process combinations in parallel
    export PROJECT_PATH MODEL_JSON_PATH
    export -f process_combination
    
    local jobs=4
    parallel --jobs "${jobs}" --progress \
        process_combination {1} {2} ::: "${cell_lines[@]}" ::: "${models[@]}"
    
    echo "Processing complete"
}

main "$@"