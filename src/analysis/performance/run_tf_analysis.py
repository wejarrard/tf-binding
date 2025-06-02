#!/usr/bin/env python3
"""
Example script showing how to use check_cell_lines_in_chip with tf_metrics
for visualization of transcription factor performance.
"""

import os
import json
import logging
import argparse
import polars as pl
from typing import List, Dict

from cell_line_utils import check_cell_lines_in_chip
from tf_metrics import SampleConfig, calculate_metrics_by_tf, plot_tf_metrics_detailed

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Analyze TF binding performance using cell line data')
    parser.add_argument('--predictions_dir', type=str, required=True,
                        help='Directory containing prediction results')
    parser.add_argument('--chip_input_dir', type=str, required=True,
                        help='Directory containing ChIP-seq BED files')
    parser.add_argument('--aligned_chip_data_dir', type=str, required=True,
                        help='Directory containing aligned ATAC data')
    parser.add_argument('--cell_line_mapping', type=str, required=True,
                        help='JSON file mapping ChIP cell line keys to full names')
    parser.add_argument('--output_plot', type=str, default='tf_metrics_dashboard.png',
                        help='Output file for the metrics visualization')
    
    args = parser.parse_args()
    
    # Load cell line mapping
    with open(args.cell_line_mapping, 'r') as f:
        cell_lines = json.load(f)
    
    # Check available cell lines
    total_cell_lines = check_cell_lines_in_chip(
        args.chip_input_dir, 
        cell_lines, 
        args.aligned_chip_data_dir
    )
    logger.info(f"Found {total_cell_lines} usable cell lines in ChIP data directory")
    
    # Load prediction results
    sample_configs = []
    dfs = []
    
    # Find all prediction files in the directory
    for filename in os.listdir(args.predictions_dir):
        if filename.endswith("_predictions.parquet"):
            # Extract TF name from filename
            tf_name = filename.split("_")[0]
            sample_name = filename.split("_")[1] if len(filename.split("_")) > 2 else "unknown"
            
            # Create sample config
            config = SampleConfig(
                label=tf_name,
                sample=sample_name,
                ground_truth_file=os.path.join(args.chip_input_dir, f"{tf_name}_binding.bed")
            )
            
            # Load predictions dataframe
            df = pl.read_parquet(os.path.join(args.predictions_dir, filename))
            
            sample_configs.append(config)
            dfs.append(df)
            
            logger.info(f"Loaded predictions for {tf_name} (sample: {sample_name})")
    
    # Calculate metrics
    metrics_df = calculate_metrics_by_tf(
        sample_configs, 
        dfs, 
        args.aligned_chip_data_dir, 
        cell_lines
    )
    
    # Create visualization
    logger.info(f"Generating metrics visualization to {args.output_plot}")
    plot_tf_metrics_detailed(metrics_df, save_path=args.output_plot)
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main() 