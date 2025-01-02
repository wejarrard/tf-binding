#!/bin/bash

# Script to process ATAC-seq data with single-sample optimization

# Check if the correct number of arguments is given
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 SAMPLE_ID BAM_STRING DIR_OUT SCRATCH_DIR"
    exit 1
fi

# Assign arguments to variables
SAMPLE_ID=$1
BAM_STRING=$2
DIR_OUT=$3
SCRATCH_DIR=$4

# Display the input arguments for verification
echo "Processing sample: $SAMPLE_ID"
echo "BAM files: $BAM_STRING"
echo "Output directory: $DIR_OUT"

# Define tool paths and parameters
R_SCRIPT_SHIFT="/home/project/ATACqseQC_shiftGAlignment.R"
CHROM_FILE="/home/project/chromosome_file.txt"
N_CORES=8
MAX_MEM_PER_CORE="4G"
GENOME_BLACKLIST="/home/project/GRCh38_unified_blacklist.bed"
REFERENCE_FA="/home/project/genome.fa"
BIN_SIZE=1

# Step 1: Create directory structure
echo "Creating directory for sample ${SAMPLE_ID}"
mkdir -p ${DIR_OUT}/${SAMPLE_ID}/{bam,peaks,QC,pileup_mod}
mkdir -p ${SCRATCH_DIR}/${SAMPLE_ID}

# Step 2: Handle BAM file(s)
# Count number of BAM files
BAM_COUNT=$(echo ${BAM_STRING} | tr ' ' '\n' | wc -l)
FN_OUT=${DIR_OUT}/${SAMPLE_ID}/bam/${SAMPLE_ID}_merge.bam

if [ ${BAM_COUNT} -gt 1 ]; then
    echo "Multiple BAM files detected (${BAM_COUNT}), merging..."
    ${SAMTOOLS} merge -@ ${N_CORES} -f ${FN_OUT} ${BAM_STRING}
else
    echo "Single BAM file detected, creating symlink..."
    ln -sf ${BAM_STRING} ${FN_OUT}
fi

# Step 3: Index the BAM file
echo "Indexing BAM file"
${SAMTOOLS} index ${FN_OUT} ${FN_OUT}.bai

# Step 4: Shift reads
echo "Shifting reads for ${SAMPLE_ID}"
FN_BAM_SHIFTED=${SCRATCH_DIR}/${SAMPLE_ID}/${SAMPLE_ID}_merge.shifted.bam
FN_BAM_SHIFTED_SORTED=${DIR_OUT}/${SAMPLE_ID}/bam/${SAMPLE_ID}_merge.sorted.nodup.shifted.bam

Rscript ${R_SCRIPT_SHIFT} \
  --sample_id ${SAMPLE_ID} \
  --dir_bam ${DIR_OUT}/${SAMPLE_ID}/bam \
  --dir_out ${SCRATCH_DIR}/${SAMPLE_ID} \
  --num_cores ${N_CORES}

${SAMTOOLS} sort \
  -@ ${N_CORES} \
  -T ${SCRATCH_DIR}/${SAMPLE_ID} \
  -m ${MAX_MEM_PER_CORE} \
  -O bam \
  -o ${FN_BAM_SHIFTED_SORTED} ${FN_BAM_SHIFTED}

${SAMTOOLS} index \
  -@ ${N_CORES} \
  ${FN_BAM_SHIFTED_SORTED} \
  ${FN_BAM_SHIFTED_SORTED}.bai

# Step 5: Generate coverage tracks (bam to bigwig)
echo "Generating coverage tracks for ${SAMPLE_ID}"
FILE_BAM=${FN_BAM_SHIFTED_SORTED}
FILE_BIGWIG=${DIR_OUT}/${SAMPLE_ID}/bam/${SAMPLE_ID}_merge.sorted.nodup.shifted.bam.bw

bamCoverage -b ${FILE_BAM} \
            -o ${FILE_BIGWIG} \
            -bl ${GENOME_BLACKLIST} \
            --effectiveGenomeSize 2913022398 \
            --normalizeUsing BPM \
            -bs ${BIN_SIZE}  \
            -p ${N_CORES}

# Step 6: Peak calling with MACS2
echo "Peak calling with MACS2 for ${SAMPLE_ID}"
DIR_PEAKS=${DIR_OUT}/${SAMPLE_ID}/peaks
FILE_BAM=${FN_BAM_SHIFTED_SORTED}

FN_BROAD_UNSORTED=${SCRATCH_DIR}/${SAMPLE_ID}_peaks.broadPeak
FN_BROAD=${DIR_PEAKS}/${SAMPLE_ID}.broadPeak
FN_BROAD_FILTERED=${DIR_PEAKS}/${SAMPLE_ID}.filtered.broadPeak
FN_NARROW_UNSORTED=${SCRATCH_DIR}/${SAMPLE_ID}_peaks.narrowPeak
FN_NARROW=${DIR_PEAKS}/${SAMPLE_ID}.narrowPeak
FN_NARROW_FILTERED=${DIR_PEAKS}/${SAMPLE_ID}.filtered.narrowPeak

# BROAD peaks
${MACS2} callpeak \
    -t ${FILE_BAM} \
    -f BAMPE \
    -g hs \
    --nomodel \
    --broad \
    --keep-dup all \
    -n ${SAMPLE_ID} \
    --outdir ${SCRATCH_DIR}/${SAMPLE_ID}

# NARROW peaks
${MACS2} callpeak \
    -t ${FILE_BAM} \
    -f BAMPE \
    -g hs \
    --nomodel \
    -B \
    --keep-dup all \
    --call-summits \
    -n ${SAMPLE_ID} \
    --outdir ${SCRATCH_DIR}/${SAMPLE_ID}

sort -k 8gr,8gr ${FN_BROAD_UNSORTED} | awk 'BEGIN{OFS="\t"}{$4="Peak_"NR; print $0}' > ${FN_BROAD}
sort -k 8gr,8gr ${FN_NARROW_UNSORTED} | awk 'BEGIN{OFS="\t"}{$4="Peak_"NR; print $0}' > ${FN_NARROW}

${BEDTOOLS} intersect -v -a ${FN_BROAD} -b ${GENOME_BLACKLIST} | \
    awk 'BEGIN{OFS="\t"}{if($5>1000)$5=1000; print $0}' | \
    grep -P 'chr[0-9XY]+(?!_)' > ${FN_BROAD_FILTERED}

${BEDTOOLS} intersect -v -a ${FN_NARROW} -b ${GENOME_BLACKLIST} | \
    awk 'BEGIN{OFS="\t"}{if($5>1000)$5=1000; print $0}' | \
    grep -P 'chr[0-9XY]+(?!_)' > ${FN_NARROW_FILTERED}

# Step 7: Generate QC metrics for peak set
echo "Generating QC metrics for peak set"
FN_BAM=${FN_BAM_SHIFTED_SORTED}
FN_PEAKS=${DIR_PEAKS}/${SAMPLE_ID}.narrowPeak
DIR_QC=${DIR_OUT}/${SAMPLE_ID}/QC
FN_OUT=${DIR_QC}/${SAMPLE_ID}.narrowPeak_ATAC_QC.txt

FN_TAG=${SCRATCH_DIR}/${SAMPLE_ID}.tagAlign
FN_READCOUNT=${SCRATCH_DIR}/${SAMPLE_ID}.n_reads
FN_PEAKCOUNT=${SCRATCH_DIR}/${SAMPLE_ID}.reads_in_peak
FN_FRIP=${SCRATCH_DIR}/${SAMPLE_ID}.FRIP
FN_SUMMARY=${SCRATCH_DIR}/${SAMPLE_ID}.FRIP.summary.txt
FN_TSSE_OUT=${SCRATCH_DIR}/${SAMPLE_ID}.TSSE

echo "Counting BAM reads"
${SAMTOOLS} view -c ${FN_BAM} > ${FN_READCOUNT}
${BEDTOOLS} bamtobed -i ${FN_BAM} | awk 'BEGIN{OFS="\t"}{$4="N"; $5="1000"; print $0}' > ${FN_TAG}
n_reads=$(head -n 1 ${FN_READCOUNT})
echo "Number of reads: ${n_reads}"

echo "Intersecting BAM with peaks"
${BEDTOOLS} sort -i ${FN_PEAKS} | ${BEDTOOLS} merge -i stdin | \
    ${BEDTOOLS} intersect -u -a ${FN_TAG} -b stdin | wc -l > ${FN_PEAKCOUNT}
n_peak=$(head -n 1 ${FN_PEAKCOUNT})
frip=$(echo "scale=4; $n_peak / $n_reads" | bc)
echo ${frip} > ${FN_FRIP}
echo "Number of intersecting peaks: ${n_peak}, FRIP: ${frip}"

echo "Calculating TSSE"
Rscript /home/project/calculate_ATAC_TSSE.R \
  --input ${FN_BAM} \
  --output ${FN_TSSE_OUT} \
  --verbose

echo -e "n_reads\t${n_reads}" > ${FN_SUMMARY}
echo -e "n_peaks\t${n_peak}" >> ${FN_SUMMARY}
echo -e "FRIP\t${frip}" >> ${FN_SUMMARY}
cat ${FN_SUMMARY} ${FN_TSSE_OUT} > ${FN_OUT}
echo "Wrote ${FN_OUT}"

rm ${FN_TAG} ${FN_PEAKCOUNT} ${FN_READCOUNT} ${FN_FRIP} ${FN_SUMMARY} ${FN_TSSE_OUT}

# Step 8: Samtools mpileup for additional QC or analysis
echo "Performing samtools mpileup for additional QC or analysis"
FN_IN=${FN_BAM_SHIFTED_SORTED}
DIR_OUT_PILEUP=${DIR_OUT}/${SAMPLE_ID}/pileup_mod
DIR_QC=${DIR_OUT}/${SAMPLE_ID}/QC
# Always regenerate the modified pileup files for chr1 and chrY
N_READS=$(awk '$1 == "n_reads" {print $2}' ${DIR_QC}/${SAMPLE_ID}.narrowPeak_ATAC_QC.txt)
echo "Number of reads: ${N_READS}"

while read line; do
    CHR=$(echo $line | awk '{print $1}')
    ${SAMTOOLS} mpileup -f ${REFERENCE_FA} \
        -r ${CHR} \
        ${FN_IN} | awk -v n_reads="${N_READS}" -F "\t" \
        '{OFS=FS; $4=$4/n_reads*1000000; print }' > ${DIR_OUT_PILEUP}/${CHR}.pileup
done < ${CHROM_FILE}

# Always compress the pileup files (overwrite if they exist)
for f in ${DIR_OUT_PILEUP}/*.pileup; do
   ${BGZIP} -@ ${N_CORES} -f "$f"  # Added -f flag to force overwrite
done

# Always index the compressed pileup files (overwrite if they exist)
for f in ${DIR_OUT_PILEUP}/*.pileup.gz; do
    ${TABIX} -s 1 -b 2 -e 2 -f ${f}  # Added -f flag to force overwrite
done

# Clean up temporary files at the end of the script
echo "Cleaning up temporary files"
rm -rf ${SCRATCH_DIR}/${SAMPLE_ID}