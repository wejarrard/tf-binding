#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import tempfile
import subprocess
from typing import Dict, List, Set
import pandas as pd
import random
# ============================================================
# Constants
# ============================================================
VALID_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
LENGTH_THRESHOLD = 16000

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

def load_atac_peaks(cell_line: str, aligned_data_dir: str) -> pd.DataFrame:
    """Load ATAC peaks for a given cell line."""
    peak_file = os.path.join(aligned_data_dir, cell_line, "peaks", f"{cell_line}.filtered.broadPeak")
    if not validate_file_exists(peak_file):
        logging.warning(f"Peak file not found for {cell_line}: {peak_file}")
        return pd.DataFrame()
    
    df = pd.read_csv(peak_file, sep="\t", header=None, names=["chr", "start", "end"], usecols=[0, 1, 2])
    df = df[(df["end"] - df["start"]) <= LENGTH_THRESHOLD]  # Filter by length
    df = df[df["chr"].isin(VALID_CHROMOSOMES)]  # Filter by chromosomes
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
                    f"-c 4 -o collapse,distinct "
                    f"-delim ',' > {output_bed}")
        subprocess.run(merge_cmd, shell=True, check=True)

        # Read the final output
        result = pd.read_csv(output_bed, sep="\t", header=None, names=["chr", "start", "end", "cell_lines"], usecols=[0, 1, 2, 3])
        
        # Process to get source cell line (random one) and all cell lines
        result["source_cell_line"] = result["cell_lines"].str.split(",").apply(lambda x: random.choice(x))
        
        # Reorder columns to match desired output format
        result = result[["chr", "start", "end", "source_cell_line", "cell_lines"]]
        
        return result

# ============================================================
# Main Function
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Generate combined ATAC peaks BED file")
    parser.add_argument("--aligned_data_dir", type=str, required=True,
                        help="Directory containing aligned ATAC data")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output BED file path")
    parser.add_argument("--cell_lines", type=str, nargs="+", required=True,
                        help="List of cell lines to process")
    
    args = parser.parse_args()
    
    # Process peaks using bedtools
        # Save to file without header
    # combined_peaks.to_csv(args.output_file, sep="\t", index=False, header=False)
    combined_peaks = process_peaks_with_bedtools(args.cell_lines, args.aligned_data_dir)
    
    # Save to file with header
    combined_peaks.to_csv(args.output_file, sep="\t", index=False)
    logging.info(f"Combined peaks saved to: {args.output_file}")
    logging.info(f"Total peaks processed: {len(combined_peaks)}")

if __name__ == "__main__":
    main() 