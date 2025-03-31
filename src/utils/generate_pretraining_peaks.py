#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import tempfile
import subprocess
from typing import Dict, List, Set, Tuple
import pandas as pd
import random
from src.training.pretraining.data import CELL_LINES

# ============================================================
# Constants
# ============================================================
VALID_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
LENGTH_THRESHOLD = 16_000

# ============================================================
# Logging Configuration
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

# ============================================================
# Utility Functions
# ============================================================
def validate_file_exists(path: str) -> bool:
    return os.path.exists(path)

def filter_by_peak_length(df: pd.DataFrame, threshold: int = LENGTH_THRESHOLD) -> pd.DataFrame:
    return df[(df["end"] - df["start"]) <= threshold]

def filter_by_chromosomes(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["chr"].isin(VALID_CHROMOSOMES)]

def load_atac_peaks(cell_line: str, aligned_data_dir: str) -> pd.DataFrame:
    """Load ATAC peaks for a given cell line."""
    peak_file = os.path.join(aligned_data_dir, cell_line, "peaks", f"{cell_line}.filtered.broadPeak")
    if not validate_file_exists(peak_file):
        logging.warning(f"Peak file not found for {cell_line}: {peak_file}")
        return pd.DataFrame()
    
    df = pd.read_csv(peak_file, sep="\t", header=None, names=["chr", "start", "end"], usecols=[0, 1, 2])
    df = filter_by_peak_length(df)
    df = filter_by_chromosomes(df)
    df["cell_line"] = cell_line  # Add cell line column
    return df.sort_values(by=["chr", "start", "end"]).reset_index(drop=True)

def process_peaks_with_bedtools(cell_lines: List[str], aligned_data_dir: str) -> pd.DataFrame:
    """Process peaks using bedtools merge."""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 1: Create individual BED files for each cell line
        merged_bed = os.path.join(temp_dir, "merged.bed")
        with open(merged_bed, 'w') as outfile:
            for cell_line in cell_lines:
                peaks_df = load_atac_peaks(cell_line, aligned_data_dir)
                if not peaks_df.empty:
                    # Write to the merged file directly
                    peaks_df.to_csv(outfile, sep="\t", index=False, header=False)
        
        # Step 2: Sort the merged file
        sorted_bed = os.path.join(temp_dir, "sorted.bed")
        sort_cmd = f"sort -k1,1 -k2,2n {merged_bed} > {sorted_bed}"
        subprocess.run(sort_cmd, shell=True, check=True)
        
        # Step 3: Use bedtools merge to combine overlapping regions and collect cell lines
        output_bed = os.path.join(temp_dir, "output.bed")
        merge_cmd = (f"bedtools merge -i {sorted_bed} "
                     f"-c 4 -o collapse "
                     f"-delim ',' > {output_bed}")
        subprocess.run(merge_cmd, shell=True, check=True)
        
        # Read the final output
        result = pd.read_csv(output_bed, sep="\t", header=None, names=["chr", "start", "end", "cell_lines"], usecols=[0, 1, 2, 3])
        
        # remove duplicates from cell_lines, then return it as
        result["cell_lines"] = result["cell_lines"].str.split(",").apply(lambda x: ",".join(set(x)))
        print(result.head())
        
        # Process to get source cell line (random one) and all cell lines
        result["source_cell_line"] = result["cell_lines"].str.split(",").apply(lambda x: random.choice(x))
        
        # Add column with the count of cell lines
        result["cell_line_count"] = result["cell_lines"].str.split(",").apply(len)
        
        # Reorder columns to match desired output format
        result = result[["chr", "start", "end", "source_cell_line", "cell_lines", "cell_line_count"]]
        
        return result

def split_train_validation(df: pd.DataFrame, validation_cell_line_threshold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into training and validation sets based on cell line count."""
    # Validation set: entries with fewer than threshold cell lines
    validation_df = df[df["cell_line_count"] < validation_cell_line_threshold].copy().sample(10_000)
    
    # Training set: all entries
    training_df = df.copy()
    
    # Remove cell_line_count column as it's not needed in final output
    validation_df = validation_df.drop(columns=["cell_line_count"])
    training_df = training_df.drop(columns=["cell_line_count"])
    
    return training_df, validation_df

# ============================================================
# Main Function
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Generate combined ATAC peaks BED file")
    parser.add_argument("--aligned_data_dir", type=str, required=True,
                        help="Directory containing aligned ATAC data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for BED files")
    parser.add_argument("--cell_lines", type=str, nargs="+", required=False, default=CELL_LINES,
                        help="List of cell lines to process")
    parser.add_argument("--validation_threshold", type=int, default=4,
                        help="Cell line count threshold for validation set (samples with fewer cell lines than this will be in validation)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define output file paths
    train_output_file = os.path.join(args.output_dir, "train.bed")
    valid_output_file = os.path.join(args.output_dir, "valid.bed")
    
    # Process peaks using bedtools
    combined_peaks = process_peaks_with_bedtools(args.cell_lines, args.aligned_data_dir)
    
    # Split data into training and validation sets
    train_df, valid_df = split_train_validation(combined_peaks, args.validation_threshold)
    
    # Save to files with header
    train_df.to_csv(train_output_file, sep="\t", index=False)
    valid_df.to_csv(valid_output_file, sep="\t", index=False)
    
    logging.info(f"Training set saved to: {train_output_file}")
    logging.info(f"Validation set saved to: {valid_output_file}")
    logging.info(f"Total peaks processed: {len(combined_peaks)}")
    logging.info(f"Training set size: {len(train_df)}")
    logging.info(f"Validation set size: {len(valid_df)}")
    logging.info(f"Validation set contains samples with fewer than {args.validation_threshold} cell lines")

if __name__ == "__main__":
    main()