#!/bin/bash

# Check if the input directory is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_directory>"
    exit 1
fi

input_dir=$1

rm /data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/qsub_scripts/*
rm /data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/qsub_logs/*

# Base directory for signal files
signal_base_dir="/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines"

# Iterate over each transcription factor directory in the input directory
for tf_dir in "$input_dir"/*/; do
    tf=$(basename "$tf_dir")

    # Generate a unique script for this TF
    script_path="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/qsub_scripts/run_tobias_${tf}.sh"
    mkdir -p "$(dirname "$script_path")"

    cat <<EOF > "$script_path"
#!/bin/bash
#$ -N tobias_$tf
#$ -pe smp 4
#$ -l mem_free=16G
#$ -o /data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/qsub_logs/tobias_$tf.out
#$ -j y
#$ -V


# Declare the associative array for cell lines
declare -A cell_lines=(
    ["293"]="HEK_293"
    ["A549"]="A549"
    ["HUH-7"]="HuH-7"
    ["K-562"]="K562"
    ["MCF-7"]="MCF7"
    ["22RV1"]="22Rv1"
    ["A-375"]="A-375"
    ["C4-2"]="C4-2"
    ["LNCAP"]="LNCAP"
    ["PC-3"]="PC-3"
    ["THP-1"]="THP-1"
    ["UM-UC-3"]="UM-UC-3"
    ["VCAP"]="VCAP"
    ["CAMA-1"]="CAMA-1"
    ["ECC-1"]="AN3_CA"
    ["HCC1954"]="HCC1954"
    ["HEC-1-A"]="HEC-1-A"
    ["ISHIKAWA"]="Ishikawa"
    ["MDA-MB-134-VI"]="MDA-MB-134-VI"
    ["MDA-MB-453"]="MDA-MB-453"
    ["NCI-H3122"]="NCI-H3122"
    ["NCI-H441"]="NCI-H441"
    ["RT4"]="RT4"
    ["SK-BR-3"]="SK-BR-3"
    ["GP5D"]="GP5D"
    ["LS-180"]="LS-180"
    ["CFPAC-1"]="CFPAC-1"
    ["GM12878"]="GM12878"
    ["MDA-MB-231"]="MDA-MB-231"
    ["HELA"]="HELA"
    ["SK-N-SH"]="SK-N-SH"
    ["HMEC"]="HMEC"
    ["U2OS"]="U2OS"
    ["BJ"]="BJ_hTERT"
    ["KELLY"]="KELLY"
    ["U-87 MG"]="U-87_MG"
    ["MV-4-11"]="MV-4-11"
    ["MM.1S"]="MM.1S"
    ["MOLT-4"]="MOLT-4"
    ["REH"]="REH"
    ["Raji"]="Raji"
    ["Ramos"]="Ramos"
    ["Hep G2"]="Hep_G2"
    ["NCI-H660"]="NCI-H660"
    ["NB69"]="NB69"
    ["SK-N-AS"]="SK-N-AS"
    ["HEC-1-B"]="HEC-1-B"
    ["HT29"]="HT29"
    ["OCI-LY8"]="OCI-LY8"
    ["DLD-1"]="DLD-1"
    ["U-937"]="U-937"
    ["IMR-90"]="IMR-90"
    ["SK-MEL-28"]="SK-MEL-28"
    ["MKN-45"]="MKN-45"
    ["AGS"]="AGS"
    ["HCT-15"]="HCT-15"
    ["SW 620"]="SW_620"
    ["HT1376"]="HT-1376"
    ["NALM-6"]="NALM-6"
    ["NAMALWA"]="NAMALWA"
    ["DOHH-2"]="DOHH-2"
    ["OCI-LY1"]="OCI-LY1"
    ["U-266"]="U-266"
    ["SUM 159PT"]="SUM-159PT"
    ["KG-1"]="KG-1"
    ["COLO 205"]="COLO_205"
    ["CAL-27"]="CAL_27"
    ["PANC-1"]="PANC-1"
    ["DU 145"]="DU_145"
    ["CAL-51"]="CAL_51"
    ["OCI-AML3"]="OCI-AML3"
    ["KARPAS-299"]="KARPAS-299"
    ["HL-60"]="HL-60"
    ["RH-41"]="RH-41"
    ["OVCAR-8"]="OVCAR-8"
    ["NTERA-2"]="NTERA-2"
    ["SW 780"]="SW_780"
    ["RD"]="RD"
    ["T24"]="T24"
    ["OVCAR-3"]="OVCAR-3"
    ["SK-N-FI"]="SK-N-FI"
    ["JURKAT"]="JURKAT"
    ["DAUDI"]="Daudi"
    ["K562"]="K562"
    ["Raji"]="Raji"
    ["NALM-6"]="NALM-6"
    ["NCI-H69"]="NCI-H69"
    ["RPMI-8226"]="RPMI_8226"
    ["NCI-H1437"]="NCI-H1437"
    ["COLO-320"]="COLO_320"
    ["LP-1"]="LP-1"
    ["NCI-H929"]="NCI-H929"
    ["OCIMY-5"]="OCI-My5"
    ["MOLT4"]="MOLT-4"
    ["RS411"]="RS411"
    ["OCI-AML3"]="OCI-AML3"
    ["OCI-LY3"]="OCI-LY3"
    ["OPM2"]="OPM-2"
    ["BEAS-2B"]="BEAS-2B"
    ["HCC70"]="HCC70"
    ["Capan-1"]="Capan-1"
    ["NTERA-2"]="NTERA-2"
    ["HCT-116"]="HCT_116"
    ["DU145"]="DU145"
    ["U-2OS"]="U-2OS"
)

echo "sourcing bashrc"
source /data1/home/wjarrard/.bashrc
echo "activating conda environment"
conda activate processing

echo "Running TOBIAS for TF: $tf"

rm -r ${input_dir}/${tf}/tobias/*

echo "cleaned directory"

mkdir ${input_dir}/${tf}/tobias/

# Iterate over each file in the {TF}/output directory
for bed_file in ${tf_dir}output/*.bed; do
    # Extract the cell line key from the filename
    cell_line_key=\$(basename "\$bed_file" | cut -d'_' -f1)

    # Check if the cell line key exists in the associative array
    if [[ -n "\${cell_lines[\$cell_line_key]}" ]]; then
        cell_line=\${cell_lines[\$cell_line_key]}

        mkdir /data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/${tf}/tobias/\${cell_line}

        python /data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/tobias_generate_samples.py $tf \$cell_line
        
        signal_file="${signal_base_dir}/\${cell_line}/bam/\${cell_line}_merge.sorted.nodup.shifted.bam.bw"
        region_file="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/${tf}/tobias/\${cell_line}/balanced.bed"
        output_file="${input_dir}/${tf}/tobias/\${cell_line}/\${cell_line}_${tf}.bw"

        # Create output directory if it doesn't exist
        mkdir -p "\$(dirname "\$output_file")"

        # Call TOBIAS FootprintScores
        TOBIAS FootprintScores --signal "\$signal_file" --regions "\$region_file" --output "\$output_file" --cores 4
        TOBIAS BINDetect \\
        --motifs /data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/HOCOMOCOv11_core_HUMAN_mono_meme_format.meme \\
        --signals \$output_file \\
        --genome /data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/genome.fa \\
        --peaks \$region_file \\
        --outdir "${input_dir}/${tf}/tobias/\${cell_line}/" \\
        --cond_names $tf \\
        --cores 4

    else
        echo "Warning: Cell line key \$cell_line_key not found in the associative array."
    fi
done
EOF

    # Make the script executable
    chmod +x "$script_path"

    # Submit the script to the cluster
    qsub "$script_path"
done
