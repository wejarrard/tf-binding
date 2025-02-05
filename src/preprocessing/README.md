# ATAC-seq Data Processing Pipeline

This pipeline processes ATAC-seq data and integrates it with transcription factor binding information.

## Prerequisites

- Create a `priority_data` folder in the ATAC directory
- Format cell line data files as `cell_line.txt` containing SRR IDs (one per line)
- Ensure `cell_line_mapping.json` is up to date in `/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/`

## Pipeline Steps

### 1. Initial Data Processing

```bash
# Pull down ATAC-seq data
./atac/pull_down.sh

# Call peaks on the data
./atac/call_peaks.sh

# Submit processing jobs
./atac/submit_jobs.sh
```

### 2. Transcription Factor Analysis

Navigate to the preprocessing chip directory:
```bash
cd tf-binding/src/preprocessing/chip
```

Before running the analysis:
1. Edit `process_tfs_qsub.py` to include your transcription factors of interest
2. Verify `cell_line_mapping.json` is current
3. Run the processing script:
```bash
python process_tfs_qsub.py
```

### 3. Data Migration

Move the processed data to the final location:
```bash
mv priority_data data
```

## File Structure

```
atac/
├── priority_data/
│   └── cell_line.txt    # SRR IDs, one per line
├── pull_down.sh
├── call_peaks.sh
└── submit_jobs.sh
```

## Notes

- Ensure all file paths are correct before running scripts
- Verify SRR IDs are properly formatted in cell line files
- Keep `cell_line_mapping.json` updated for accurate results