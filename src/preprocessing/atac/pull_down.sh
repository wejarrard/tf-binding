#!/bin/bash

# pull_down.sh
conda activate processing

# Define base directories
base_dir=/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/preprocessing/atac
dir_sratool=/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/preprocessing/atac/sratoolkit.3.1.1-ubuntu64/bin
output_dir=/data1/datasets_1/human_cistrome/SRA

# Create logs directory if it doesn't exist
mkdir -p ${base_dir}/logs

# Generate metadata
python generate_metadata.py ${base_dir}/priority_data

# Loop through each file in the priority_data directory
for fn_srr_id in ${base_dir}/priority_data/*.txt; do
    # Extract the file name without path and extension
    file_name=$(basename "$fn_srr_id" .txt)
    
    # Start a new screen session with the processing command directly
    screen -dmS "screen_${file_name}" bash -c "
        # Read and trim the SRR ID
        srr_id=\$(tr -d '[:space:]' < '${fn_srr_id}')
        
        if [ -n \"\${srr_id}\" ]; then
            echo \"Processing SRR: \${srr_id}\"
            ${dir_sratool}/fasterq-dump.3.1.1 \"\${srr_id}\" -O \"${output_dir}\" --split-3
            echo \"Completed processing \${srr_id}\"
        else
            echo \"No valid SRR ID found in ${file_name}\"
        fi
        
        echo \"Finished processing ${file_name}\"
    " > "${base_dir}/logs/screen_${file_name}.log" 2>&1
done

# first create priority_data

# 1. /data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/preprocessing/atac/pull_down.sh

# 2. call_peaks.sh in this directory

# 3. call /data1/projects/human_cistrome/aligned_chip_data/SRA/submit_jobs.sh CELL_LINES (which are listed in priority_data if you remove the .txt extension)

# 4. then go to /data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/bed_file_aggregation and run the following command:
# edit the python process_tfs_qsub.py to include the transcription factors of interest
# make sure /data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/cell_line_mapping.json is updated
# then runpython process_tfs_qsub.py

# 5. move priority_data to data
