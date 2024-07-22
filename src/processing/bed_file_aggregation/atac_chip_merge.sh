#!/bin/bash

# Define an array of cell lines and corresponding transcription factors
transcription_factors=("FOXM1")
output_dir=./data/FOXM1/output
# Define the directory with BED files
bed_dir="./data/FOXM1/final/"

# Define transcription factors
declare -A cell_lines

cell_lines["293"]="HEK_293"
cell_lines["A549"]="A549"
cell_lines["HUH-7"]="HuH-7"
cell_lines["K-562"]="K562"
cell_lines["MCF-7"]="MCF7"
cell_lines["22RV1"]="22Rv1"
cell_lines["A-375"]="A-375"
cell_lines["C4-2"]="C4-2"
cell_lines["LNCAP"]="LNCAP"
cell_lines["PC-3"]="PC-3"
cell_lines["THP-1"]="THP-1"
cell_lines["UM-UC-3"]="UM-UC-3"
cell_lines["VCAP"]="VCAP"
cell_lines["CAMA-1"]="CAMA-1"
cell_lines["ECC-1"]="AN3_CA"
cell_lines["HCC1954"]="HCC1954"
cell_lines["HEC-1-A"]="HEC-1-A"
cell_lines["ISHIKAWA"]="Ishikawa"
cell_lines["MDA-MB-134-VI"]="MDA-MB-134-VI"
cell_lines["MDA-MB-453"]="MDA-MB-453"
cell_lines["NCI-H3122"]="NCI-H3122"
cell_lines["NCI-H441"]="NCI-H441"
cell_lines["RT4"]="RT4"
cell_lines["SK-BR-3"]="SK-BR-3"
cell_lines["GP5D"]="GP5D"
cell_lines["LS-180"]="LS-180"
cell_lines["CFPAC-1"]="CFPAC-1"
cell_lines["GM12878"]="GM12878"
cell_lines["MDA-MB-231"]="MDA-MB-231"
cell_lines["HELA"]="HELA"
cell_lines["SK-N-SH"]="SK-N-SH"
cell_lines["HMEC"]="HMEC"
cell_lines["U2OS"]="U2OS"

# Create the output directory if it does not exist
mkdir -p "$output_dir"

# List available BED files in the directory
available_bed_files=($(ls $bed_dir))

# Loop through each cell line and transcription factor
for cell_line_key in "${!cell_lines[@]}"; do
    for tf in "${transcription_factors[@]}"; do
        bed_file_name="${cell_line_key}_${tf}_merged.bed"
        # Check if the bed file exists in the directory
        if [[ " ${available_bed_files[@]} " =~ " ${bed_file_name} " ]]; then
            # Define paths to the input files
            bed_file1="${bed_dir}${bed_file_name}"
            bed_file2="/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/${cell_lines[$cell_line_key]}/peaks/${cell_lines[$cell_line_key]}.filtered.broadPeak"

            # Output path
            output_bed="${output_dir}/${cell_line_key}_${tf}.bed"

            # Sort both BED files before processing
            sort -k1,1 -k2,2n $bed_file1 > sorted_bed_file1.bed
            sort -k1,1 -k2,2n $bed_file2 > sorted_bed_file2.bed

            # Intersect the two sorted BED files and prepare for merging
            bedtools intersect -a sorted_bed_file1.bed \
                               -b <(awk 'OFS="\t" {print $1,$2,$3,"0"}' sorted_bed_file2.bed) \
                               -wa -wb | \
            awk 'OFS="\t" {print $1, $2, $3, $4+$8}' > $output_bed

            # Clean up intermediate files
            rm sorted_bed_file1.bed sorted_bed_file2.bed

            echo "Output available in $output_bed"
        fi
    done
done

# Remove empty files
find "$output_dir" -type f -empty -delete