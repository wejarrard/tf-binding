#!/bin/bash
# submit_jobs.sh

BASE_DIR="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/preprocessing/atac"

# Path to the processing script
PROCESS_SCRIPT="${BASE_DIR}/process_cell_line.sh"

# Create logs directory if it doesn't exist
mkdir -p "${BASE_DIR}/logs_merge_cell_lines"

# Loop over each .txt file in priority_data directory
for FILE in "${BASE_DIR}/priority_data"/*.txt; do
    # Extract cell line name by removing .txt extension and path
    CELL_LINE=$(basename "$FILE" .txt)
    
    # Construct job name and log file name
    JOB_NAME="merge_${CELL_LINE}"
    LOG_FILE="${BASE_DIR}/logs_merge_cell_lines/${JOB_NAME}.log"

    rm -f "$LOG_FILE"

    # Submit the job to SGE
    qsub -cwd -N "$JOB_NAME" -j y -o "$LOG_FILE" -pe smp 8 -l mem_free=16G -l h='node4|node5|node6|node7|node8' \
    "$PROCESS_SCRIPT" "$CELL_LINE"
done