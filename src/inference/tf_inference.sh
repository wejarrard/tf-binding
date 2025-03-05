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
    local log_file="${log_dir}/${cell_line##*/}_${model}.log"
    mkdir -p "$log_dir"

    # Clean up existing output files and directories
    echo "Cleaning up existing output files and directories..."
    
    # Remove existing log file
    if [ -f "$log_file" ]; then
        rm "$log_file"
        echo "Removed existing log file: $log_file"
    fi
    
    # Extract TF name and set input file name (moved up for cleanup)
    local tf_name=$(echo "$model" | cut -d'-' -f1)
    local input_file="${model}_${cell_line##*/}"
    

    ############################################################
    # Remove existing data splits file
    local data_splits_file="${PROJECT_PATH}/data/data_splits/${input_file}.csv"
    if [ -f "$data_splits_file" ]; then
        rm "$data_splits_file"
        echo "Removed existing data splits file: $data_splits_file"
    fi
    
    # Remove existing jsonl directory
    local jsonl_dir="${PROJECT_PATH}/data/jsonl/${input_file}"
    if [ -d "$jsonl_dir" ]; then
        rm -rf "$jsonl_dir"
        echo "Removed existing jsonl directory: $jsonl_dir"
    fi

    ############################################################
    
    # Log start of processing
    {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting processing: ${cell_line##*/} - ${model}"
        echo "SRR_VALUE: ${srr_value}"
        echo "Processing TF: ${model%%-*}"
    } >> "$log_file"
    
    # Get input bed file path
    local bed_path
    if [[ "$cell_line" =~ ^LuCaP ]]; then
        bed_path="/data1/datasets_1/human_prostate_PDX/processed/ATAC/${cell_line}/${srr_value}/peaks/${srr_value}.filtered.broadPeak"
    elif [[ ! "$cell_line" =~ / ]]; then
        bed_path="/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/${cell_line}/peaks/${cell_line}.filtered.broadPeak"
    else
        bed_path="${cell_line}/peaks/${cell_line##*/}.filtered.broadPeak"
    fi
    
    # Validate input file
    if [ ! -f "${bed_path}" ]; then
        echo "Error: Input bed file ${bed_path} not found" | tee -a "$log_file"
        return 1
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
    # else if cell line doesnt have a slash in it
    elif [[ ! "$cell_line" =~ / ]]; then
        cell_line_dir="/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/${cell_line}"
    else
        cell_line_dir="${cell_line}"
        # cell line is everything after the last slash
        cell_line="${cell_line##*/}"
    fi
    
 #########################
    # Prepare data using qsub
    echo "Preparing data for ${cell_line} - ${model} using qsub..."
    local qsub_script="${log_dir}/${cell_line}_${model}.sh"
    cat << EOF > "${qsub_script}"
#!/bin/bash
#$ -N prep_${cell_line}_${model}
#$ -o ${log_dir}/${cell_line}_${model}.log
#$ -j y
#$ -l h_vmem=32G

source ~/.bashrc

source activate processing

python "${PROJECT_PATH}/src/inference/prepare_data.py" \
    --input_file "${input_file}.csv" \
    --output_path "${PROJECT_PATH}/data/jsonl/${input_file}" \
    --cell_line_dir "${cell_line_dir}"
EOF

    chmod +x "${qsub_script}"
    local job_id=$(qsub "${qsub_script}" | cut -d' ' -f3)
    
    # Wait for the job to complete
    echo "Waiting for prepare_data.py job (${job_id}) to complete..." | tee -a "$log_file"
    qstat -j "${job_id}" 2>/dev/null
    while [ $? -eq 0 ]; do
        sleep 30
        qstat -j "${job_id}" 2>/dev/null
    done

#########################
    
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
    # /data1/datasets_1/human_prostate_PDX/processed/ATAC
    # local cell_lines=("LuCaP_81:SRR12455442" "LuCaP_78:SRR12455441" "LuCaP_77CR:SRR12455440" "LuCaP_77:SRR12455439") # AR HOXB13 FOXA1
    # local cell_lines=("LuCaP_145_1:SRR12455430" "LuCaP_93:SRR12455443" "LuCaP_49:SRR12455437") #ASCL1 alts: "LuCaP_145_1:SRR12455431" "LuCaP_93:SRR12455444"
    # local cell_lines=("LuCaP_173_1:SRR12455433") #NEUROD1 alts: "LuCaP_173_1:SRR12455434"
    # local cell_lines=("LuCaP_49:SRR12455437" "LuCaP_93:SRR12455443" "LuCaP_145_1:SRR12455430" "LuCaP_145_2:SRR12455432" "LuCaP_173_1:SRR12455434") # More FOXA1 all these have replicates
    # local cell_lines=("42D-ENZR")
    local cell_lines=("/data1/projects/human_cistrome/aligned_chip_data/SRA/DF_primary/DF_2480/processed/SRR11856442")
    
    local models=("FOXA1" "HOXB13")
    
    # Process combinations in parallel
    export PROJECT_PATH MODEL_JSON_PATH
    export -f process_combination
    
    local jobs=5
    parallel --jobs "${jobs}" --progress \
        process_combination {1} {2} ::: "${cell_lines[@]}" ::: "${models[@]}"
    
    echo "Processing complete"
}

main "$@"