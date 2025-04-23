#!/bin/bash

##set -euo pipefail

# Configuration
PROJECT_PATH="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"
MODEL_JSON_PATH="${PROJECT_PATH}/src/inference/models.json"

# Function to print usage
print_usage() {
    echo "Usage: $0 --atac_dir <atac_directory> --models <model1,model2,...> [--parallel]"
    echo
    echo "Options:"
    echo "  --atac_dir   Directory containing ATAC data or cell line identifier"
    echo "  --models     Comma-separated list of models to run (specified in models.json)"
    echo "  --parallel   Optional: Run in parallel mode"
    echo
    echo "Example:"
    echo "  $0 --atac_dir /data1/datasets_1/human_prostate_PDX/processed/ATAC/LuCaP_145_1:SRR12455430 --models FOXA1,FOXA2 --parallel"
}

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
    echo "cell_line: ${cell_line} srr_value: ${srr_value}"
    if [[ "$cell_line" == LuCaP* ]]; then
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
    echo "Preparing data for ${cell_line} - ${model}..."
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

    echo "cell_line_dir: ${cell_line_dir}"

    echo "cell_line: ${cell_line}"
    
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

conda activate pterodactyl

python "${PROJECT_PATH}/src/inference/prepare_data.py" \
    --input_file "${input_file}.csv" \
    --output_path "${PROJECT_PATH}/data/jsonl/${input_file}" \
    --cell_line_dir "${cell_line_dir}"

EOF

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
    # Parse command line arguments
    local atac_dir=""
    local models=""
    local run_parallel="false"
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --atac_dir)
                atac_dir="$2"
                shift 2
                ;;
            --models)
                models="$2"
                shift 2
                ;;
            --parallel)
                run_parallel="true"
                # Check if the next argument is a flag (starts with --)
                if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
                    # If explicitly set to false, respect that
                    if [[ "$2" == "false" ]]; then
                        run_parallel="false"
                    fi
                    shift
                fi
                shift
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                echo "Error: Unknown option $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    # Validate required arguments
    if [[ -z "$atac_dir" ]]; then
        echo "Error: --atac_dir is required"
        print_usage
        exit 1
    fi
    
    if [[ -z "$models" ]]; then
        echo "Error: --models is required"
        print_usage
        exit 1
    fi
    
    # Activate conda environment
    if ! command -v conda &> /dev/null; then
        echo "Error: conda not found"
        exit 1
    fi
    

    
    # Parse cell lines from atac_dir
    local cell_lines=()
    
    if [[ "$atac_dir" == *","* ]]; then
        # If it's not a directory, treat it as a comma-separated list
        IFS=',' read -ra cell_lines <<< "$atac_dir"
        
        # Ensure we're properly handling comma-separated input that might include SRR values
        # This prevents issues if a cell line entry contains a colon
        for i in "${!cell_lines[@]}"; do
            cell_lines[$i]=$(echo "${cell_lines[$i]}" | xargs)  # Trim whitespace
        done
    else
        cell_lines=("$atac_dir")
    fi
    
    # Parse models
    local model_list=()
    IFS=',' read -ra model_list <<< "$models"
    
    # Export variables for parallel execution
    export PROJECT_PATH MODEL_JSON_PATH
    export -f process_combination
    
    # Process combinations based on parallel flag
    if [[ "$run_parallel" == "true" ]]; then
        # Process in parallel using GNU parallel
        local jobs=5
        echo "Running in parallel mode with $jobs concurrent jobs..."
        parallel --jobs "${jobs}" --progress \
            process_combination {1} {2} ::: "${cell_lines[@]}" ::: "${model_list[@]}"
    else
        # Process sequentially
        echo "Running in sequential mode..."
        for cell_line in "${cell_lines[@]}"; do
            for model in "${model_list[@]}"; do
                echo "Processing combination: $cell_line - $model"
                process_combination "$cell_line" "$model"
            done
        done
    fi
    
    echo "Processing complete"
}

# Call main function with all script arguments
main "$@"



# /data1/datasets_1/human_prostate_PDX/processed/ATAC
# local cell_lines=("LuCaP_81:SRR12455442" "LuCaP_78:SRR12455441" "LuCaP_77CR:SRR12455440" "LuCaP_77:SRR12455439") # AR HOXB13 FOXA1
# local cell_lines=("LuCaP_145_1:SRR12455430" "LuCaP_93:SRR12455443" "LuCaP_49:SRR12455437") #ASCL1 alts: "LuCaP_145_1:SRR12455431" "LuCaP_93:SRR12455444"
# local cell_lines=("LuCaP_173_1:SRR12455433") #NEUROD1 alts: "LuCaP_173_1:SRR12455434"
# local cell_lines=("LuCaP_49:SRR12455437" "LuCaP_93:SRR12455443" "LuCaP_145_1:SRR12455430" "LuCaP_145_2:SRR12455432" "LuCaP_173_1:SRR12455434") # More FOXA1 all these have replicates
# local cell_lines=("42D-ENZR")
# local cell_lines=("22Rv1")