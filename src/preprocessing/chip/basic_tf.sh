#!/bin/bash

# Usage: ./combined_script.sh TF

# Validate input
if [ $# -lt 1 ]; then
    echo "Usage: $0 TF"
    echo "Please provide one transcription factor."
    exit 1
fi

# Set transcription factor from command line argument
tf="$1"


# Path to the JSON file that maps cell line names
json_file="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/utils/cell_line_mapping.json"

# Path to the input file that maps cell lines to experiment IDs
input_file="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/2025/2025_02_07_ChIP-atlas_experimentList_parsed.tab"

# path to bed file aggregate
bed_file="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/2025/allPeaks_light.hg38.05.bed.gz"

# path to output directory for the TF
output_dir="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/transcription_factors/${tf}"


# Function to load the JSON file and create an associative array
declare -A cell_lines
load_cell_lines() {
    local key value
    # Use jq to parse the JSON and loop through keys and values
    while IFS="=" read -r key value; do
        cell_lines["$key"]="$value"
    done < <(jq -r 'to_entries | .[] | "\(.key)=\(.value)"' "$json_file")
}

# Call the function to load cell lines into the associative array
load_cell_lines

# Directory for output data
mkdir -p "$output_dir"
output_file="$output_dir/output.tsv"
filtered_output_file="$output_dir/filtered_output.tsv"

# Prepare header for output file
echo "TF\tCell Line\tExperiment ID" > "$output_file"

# Function to extract and process TF data
extract_tf_data() {
    local tf="$1"
    awk -v tf="$tf" -F '\t' '($2 == tf) {print $2, $3, $1}' OFS='\t' "$input_file"
}

# Function to extract unique cell lines for a given TF
extract_cell_lines() {
    local tf="$1"
    awk -F'\t' -v tf="$tf" '$1 == tf {print $2}' "$output_file" | sort | uniq
}

# Function to save filtered records
save_filtered_records() {
    local cell_lines="$1"
    awk -F'\t' -v tf="$tf" -v cell_lines="$cell_lines" 'BEGIN {
        split(cell_lines, arr, " ");
    }
    ($1 == tf) {
        for (i in arr) {
            if ($2 == arr[i]) {
                print $0;
            }
        }
    }' "$output_file" > "$filtered_output_file"
}


# Main processing steps
extract_tf_data "$tf" >> "$output_file"

# Extract cell lines
cell_lines=$(extract_cell_lines "$tf")

# Save filtered records
save_filtered_records "$cell_lines"

# Run Python script to aggregate experiments
echo "Aggregating records"
python aggregate_experiments.py ${output_dir} 

# Start processing the BED file
input_bed_file="${output_dir}/aggregate.tsv"

mkdir -p ${output_dir}/split ${output_dir}/merged

# Export variables to be accessible by parallel
export bed_file output_dir

# Define a function to process each line
process_line() {
    cell_line=$1
    tf=$2
    identifiers=$3

    echo "Processing: Cell Line=$cell_line, TF=$tf"

    # Create a filename based on cell line and TF
    filtered_filename="${cell_line}_${tf}_filtered.bed"
    filtered_bed_file="${output_dir}/split/${filtered_filename}"

    echo "Creating filtered bed file: $filtered_bed_file"
    
    # Create a single regex pattern from all identifiers
    pattern=$(echo "$identifiers" | tr ',' '|')
    
    # Process the BED file in a single pass
    echo "Filtering BED file with pattern..."
    zcat "$bed_file" | awk -v pattern="$pattern" '$0 ~ pattern' > "$filtered_bed_file"
    
    echo "Filtered bed file size: $(wc -l < $filtered_bed_file) lines"

    # Define intermediate and output file paths
    output_bed="${output_dir}/merged/${cell_line}_${tf}_merged.bed"

    # Process the filtered BED file - combine all steps into one awk command
    echo "Processing and merging BED file..."
    awk '{OFS="\t"; if ($1 ~ /^chr([1-9]|1[0-9]|2[0-4]|X|Y)$/) {$5=1; print}}' "$filtered_bed_file" | \
    sort -k1,1 -k2,2n | \
    bedtools merge -i - -c 5 -o sum > "$output_bed"
    
    echo "Final merged bed file size: $(wc -l < "$output_bed") lines"
    echo "Output available in $output_bed"
    echo "----------------------------------------"
}

# Export the function so parallel can use it
export -f process_line

# Use parallel to process each line from the input file, sending the correct fields to the function
cat $input_bed_file | parallel --colsep '\t' process_line {1} {2} {3}

# Non-parallel version for debugging
# echo "Running non-parallel version for debugging:"
# while IFS=$'\t' read -r cell_line tf identifiers; do
#     process_line "$cell_line" "$tf" "$identifiers"
# done < "$input_bed_file"

# Remove empty files
find "$output_dir" -type f -empty -delete
