#!/bin/bash

# Check if CELL_LINE is provided
if [ -z "$1" ]; then
    echo "Usage: $0 CELL_LINE"
    exit 1
fi

CELL_LINE="$1"

# Base directory containing the processed data
BASE_DIR="/data1/projects/human_cistrome/aligned_chip_data/SRA/$CELL_LINE/processed"

# Array to hold the BAM file paths
BAM_FILES=()

# Loop through each SRR directory and find the desired BAM file
for SRR_DIR in "$BASE_DIR"/SRR*; do
    if [ -d "$SRR_DIR" ]; then
        ALIGNMENT_DIR="$SRR_DIR/alignment"
        if [ -d "$ALIGNMENT_DIR" ]; then
            BAM_FILE="$ALIGNMENT_DIR/$(basename "$SRR_DIR").bowtie.sorted.nodup.bam"
            if [ -f "$BAM_FILE" ]; then
                BAM_FILES+=("$BAM_FILE")
            else
                echo "Warning: BAM file not found in $ALIGNMENT_DIR"
            fi
        else
            echo "Warning: Alignment directory not found in $SRR_DIR"
        fi
    else
        echo "Warning: SRR directory not found: $SRR_DIR"
    fi
done

# Check if any BAM files were found
if [ ${#BAM_FILES[@]} -eq 0 ]; then
    echo "Error: No BAM files found for CELL_LINE $CELL_LINE"
    exit 1
fi


######################################################
######################################################

# echo "Installing Docker container"

# # Define the image reference pattern
# IMAGE_PATTERN='atac_bam_merge_downstream*'
# IMAGE_TAR='/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/preprocessing/image.tar'

# # Stop all running containers
# RUNNING_CONTAINERS=$(docker ps -q)
# if [ -n "$RUNNING_CONTAINERS" ]; then
#     echo "Stopping all running containers."
#     docker stop $RUNNING_CONTAINERS
#     if [ $? -eq 0 ]; then
#         echo "Successfully stopped running containers."
#     else
#         echo "Failed to stop running containers."
#         exit 1
#     fi
# else
#     echo "No running containers found."
# fi

# # Get the list of image IDs matching the pattern
# IMAGES=$(docker images --filter=reference="$IMAGE_PATTERN" -q)

# if [ -z "$IMAGES" ]; then
#     echo "No matching images found. Proceeding to load the image."
#     docker load -i "$IMAGE_TAR"
# else
#     echo "Found matching images. Attempting to remove them."
#     docker rmi $IMAGES
#     if [ $? -eq 0 ]; then
#         echo "Successfully removed images. Proceeding to load the image."
#         docker load -i "$IMAGE_TAR"
#     else
#         echo "Failed to remove images. Not loading the image."
#     fi
# fi

# docker load -i "$IMAGE_TAR"

######################################################
######################################################
# Set variables for Docker run
SAMPLE_ID="$CELL_LINE"

# Set common Docker run parameters
CPUS=8
MEMORY=128g
OUTPUT_DIR="/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines"
SCRATCH_DIR="/scratch"

# Start building the Docker command
DOCKER_CMD="docker run --cpus=$CPUS --memory=$MEMORY"

# Set the TMPDIR environment variable inside the container
DOCKER_CMD="$DOCKER_CMD -e TMPDIR=/output/$SAMPLE_ID/tmp"

# Mount the project directory
DOCKER_CMD="$DOCKER_CMD -v /data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/preprocessing:/home/project"

# Iterate over all BAM files and add volume mounts
for BAM_FILE in "${BAM_FILES[@]}"
do
    # Get the basename of the BAM file to use inside the container
    BAM_BASENAME=$(basename "$BAM_FILE")

    # Add the volume mount to the Docker command
    DOCKER_CMD="$DOCKER_CMD -v $BAM_FILE:/data/$BAM_BASENAME"
done

# Add the output volume mount
DOCKER_CMD="$DOCKER_CMD -v $OUTPUT_DIR:/output -v $SCRATCH_DIR:/scratch" 

# Add the command to execute inside the container
DOCKER_CMD="$DOCKER_CMD atac_bam_merge_downstream $SAMPLE_ID \"/data/*\" \"/output\" \"/scratch\""

# Echo the full Docker command (for debugging)
echo "Running command: $DOCKER_CMD"

# Execute the Docker command
eval $DOCKER_CMD