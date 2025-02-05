#!/bin/bash
# call_peaks.sh

# Need to do twice in case we are not in base
conda deactivate
conda deactivate

# Set the path to the metadata directory
METADATA_DIR=/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/preprocessing/atac/metadata

# Path to the genome blacklist file
GENOME_BLACKLIST=/data1/opt/reference/Homo_sapiens/NCBI/GRCh38Decoy/Annotation/mapping/GRCh38_unified_blacklist.bed

# Iterate over each .tsv file in the metadata directory
for FN_METADATA in "${METADATA_DIR}"/*.tsv; do
    # Extract the cell line name from the filename
    CELL_LINE=$(basename "${FN_METADATA}" .tsv)

    # Define directories based on the cell line
    DIR_IN=/data1/datasets_1/human_cistrome/SRA
    DIR_BASE=/data1/projects/human_cistrome/aligned_chip_data/SRA/"${CELL_LINE}"
    DIR_OUT=/data1/projects/human_cistrome/aligned_chip_data/SRA/"${CELL_LINE}"/processed
    DIR_QC="${DIR_OUT}"/QC
    RUN_ID="${CELL_LINE}"_ATAC

    # Create necessary directories
    mkdir -p "${DIR_BASE}" "${DIR_OUT}" "${DIR_QC}" "${DIR_OUT}"/scripts "${DIR_OUT}"/logs

    # Generate the workflow
    python /data1/marlowe/marlowe/generate_cluster_workflow.py \
        --workflow_name atacseq_peakcall_paired_end_fastp_2 \
        --metadata_file "${FN_METADATA}" \
        --workflow_variables "SAMPLE_ID,SAMPLE_ID;FASTQ_1,FASTQ_1;FASTQ_2,FASTQ_2" \
        --constants "DIR_QC,${DIR_QC};PEAK_QVALUE,0.001;DIR_BASE,${DIR_BASE};DIR_IN,${DIR_IN};DIR_OUT,${DIR_OUT};GENOME_BLACKLIST,${GENOME_BLACKLIST}" \
        --dir_out "${DIR_OUT}" \
        --json_file "${RUN_ID}".json \
        --n_cores 4 \
        --mem_per_core 16 \
        --mem_monolithic 64 \
        --run_identifier "${RUN_ID}"

    # Process the workflow
    python /data1/marlowe/marlowe/process_workflow.py -v --fn_config "${DIR_OUT}"/"${RUN_ID}".json
done