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
input_file="/data1/datasets_1/human_cistrome/chip-atlas/2024_04_10_ChIP-atlas_experimentList_parsed.tab"

# path to bed file aggregate
bed_file="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/2023/allPeaks_light.hg38.05.bed"

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
python aggregate_experiments.py ${output_dir}  # Assume outputs to ./data/temp/aggregate.tsv

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

    # Create a filename based on cell line and TF
    filtered_filename="${cell_line}_${tf}_filtered.bed"
    filtered_bed_file="${output_dir}/split/${filtered_filename}"

    # Check if the filtered BED file already exists
    if [ -f "$filtered_bed_file" ]; then
        echo "Filtered BED file $filtered_bed_file already exists, skipping filtering."
    else
        touch $filtered_bed_file
        # Convert identifiers to a newline-separated list and use grep to filter the BED file
        echo "$identifiers" | tr ',' '\n' | while read identifier; do
            grep "$identifier" $bed_file >> $filtered_bed_file
        done
    fi

    # Define intermediate and output file paths
    modified_output_bed="${output_dir}/split/${cell_line}_${tf}_modified_output.bed"
    sorted_output_bed="${output_dir}/split/${cell_line}_${tf}_sorted_output.bed"
    output_bed="${output_dir}/merged/${cell_line}_${tf}_merged.bed"

    # Check if the output file already exists
    if [ -f "$output_bed" ]; then
        echo "Output file $output_bed already exists, skipping processing."
        return
    fi

    # Process the filtered BED file
    # Step 1: Modify the last column to be 1 and drop unwanted chromosomes
    awk '{OFS="\t"; if ($1 ~ /^chr([1-9]|1[0-9]|2[0-4]|X|Y)$/) {$5=1; print}}' $filtered_bed_file > $modified_output_bed

    # Step 2: Sort the modified BED file
    sort -k1,1 -k2,2n $modified_output_bed > $sorted_output_bed

    # Step 3: Merge intervals with bedtools and sum the last column
    bedtools merge -i $sorted_output_bed -c 5 -o sum > $output_bed

    # Optionally, remove intermediate files
    rm $modified_output_bed $sorted_output_bed

    echo "Output available in $output_bed"
}

# Export the function so parallel can use it
export -f process_line

# Use parallel to process each line from the input file, sending the correct fields to the function
cat $input_bed_file | parallel --colsep '\t' process_line {1} {2} {3}

# Define the directory with BED files
bed_dir="${output_dir}/merged/"

# Create the output directory if it does not exist
mkdir -p "$output_dir"
mkdir -p "${output_dir}/output"

# List available BED files in the directory
available_bed_files=($(ls $bed_dir))
# Loop through each cell line and transcription factor
for cell_line_key in "${!cell_lines[@]}"; do
    bed_file_name="${cell_line_key}_${tf}_merged.bed"
    # Check if the bed file exists in the directory
    if [[ " ${available_bed_files[@]} " =~ " ${bed_file_name} " ]]; then
        # Define paths to the input files
        bed_file1="${bed_dir}${bed_file_name}"
        bed_file2="/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/${cell_lines[$cell_line_key]}/peaks/${cell_lines[$cell_line_key]}.filtered.broadPeak"

        # Output path
        output_bed="${output_dir}/output/${cell_line_key}_${tf}.bed"

        # Create TF and cell line specific sorted BED file names
        sorted_bed_file1="sorted_${cell_line_key}_${tf}_file1.bed"
        sorted_bed_file2="sorted_${cell_line_key}_${tf}_file2.bed"

        # Sort both BED files before processing
        sort -k1,1 -k2,2n $bed_file1 > $sorted_bed_file1
        sort -k1,1 -k2,2n $bed_file2 > $sorted_bed_file2

        # Intersect the two sorted BED files and prepare for merging
        bedtools intersect -a $sorted_bed_file1 \
                           -b <(awk 'OFS="\t" {print $1,$2,$3,"0"}' $sorted_bed_file2) \
                           -wa -wb | \
        awk 'OFS="\t" {print $1, $2, $3, $4+$8}' > $output_bed

        # Clean up intermediate files
        rm $sorted_bed_file1 $sorted_bed_file2

        echo "Output available in $output_bed"
    fi
done


# Remove empty files
find "$output_dir" -type f -empty -delete
