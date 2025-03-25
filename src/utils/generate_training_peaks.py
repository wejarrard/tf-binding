#!/usr/bin/env python3
import os
import sys
import json
import logging
import argparse
import subprocess
import tempfile
from enum import Enum

import pandas as pd

# ============================================================
# Constants
# ============================================================
VALID_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
LENGTH_THRESHOLD = 4000
DEFAULT_VALIDATION_SPLIT = 0.2
HIGH_COUNT_QUANTILE = 0.75
MAX_COUNT_THRESHOLD = 30
MID_COUNT_THRESHOLD = 10
NO_GROUND_TRUTH_LABEL = -1
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

# ============================================================
# Enums for More Descriptive Configuration
# ============================================================
class BalanceOption(Enum):
    BALANCE_LABELS = "balance_labels"
    NO_BALANCE = "no_balance"

class RegionFilterOption(Enum):
    ALL_REGIONS = "all_regions"

class DataFilterOption(Enum):
    FILTER_DATA = "filter_data"
    NO_FILTER = "no_filter"

class LabelOption(Enum):
    POSITIVE_ONLY = "positive_only"
    BOTH_LABELS = "both_labels"

class ChipDataOption(Enum):
    CHIP_PROVIDED = "chip_provided"
    NO_CHIP = "no_chip"

class GroundTruthOption(Enum):
    NO_GROUND_TRUTH = "no_ground_truth"
    WITH_GROUND_TRUTH = "with_ground_truth"

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

def filter_by_chromosomes(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["chr"].isin(VALID_CHROMOSOMES)]

def filter_by_peak_length(df: pd.DataFrame, threshold: int = LENGTH_THRESHOLD) -> pd.DataFrame:
    return df[(df["end"] - df["start"]) <= threshold]

def drop_duplicates_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().sort_values(by=["chr", "start", "end"]).reset_index(drop=True)

def run_bedtools_command(command: str) -> None:
    subprocess.run(command, shell=True, check=True)

# ============================================================
# BED File Operations
# ============================================================
def subtract_bed_files(main_df: pd.DataFrame, subtract_df: pd.DataFrame) -> pd.DataFrame:
    col_names = main_df.columns.tolist()
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as main_file, \
         tempfile.NamedTemporaryFile(delete=False, mode='w') as sub_file, \
         tempfile.NamedTemporaryFile(delete=False, mode='w') as result_file:
        
        main_path = main_file.name
        sub_path = sub_file.name
        result_path = result_file.name

        main_df.to_csv(main_path, sep="\t", index=False, header=False)
        subtract_df.to_csv(sub_path, sep="\t", index=False, header=False)

        command = f"bedtools subtract -a {main_path} -b {sub_path} -A > {result_path}"
        run_bedtools_command(command)

        subtracted_df = pd.read_csv(result_path, sep="\t", header=None, names=col_names)
        
    os.remove(main_path)
    os.remove(sub_path)
    os.remove(result_path)

    return drop_duplicates_and_sort(subtracted_df)

def intersect_bed_files(main_df: pd.DataFrame, intersect_df: pd.DataFrame) -> pd.DataFrame:
    col_names = main_df.columns.tolist()
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as main_file, \
         tempfile.NamedTemporaryFile(delete=False, mode='w') as intersect_file, \
         tempfile.NamedTemporaryFile(delete=False, mode='w') as result_file:
        
        main_path = main_file.name
        intersect_path = intersect_file.name
        result_path = result_file.name

        main_df.to_csv(main_path, sep="\t", index=False, header=False)
        intersect_df.to_csv(intersect_path, sep="\t", index=False, header=False)

        command = f"bedtools intersect -a {main_path} -b {intersect_path} -wa -wb > {result_path}"
        run_bedtools_command(command)

        # Simply specify the exact columns we want: 0, 1, 2, and 6
        intersected_df = pd.read_csv(result_path, sep="\t", header=None, usecols=[0, 1, 2, 6], names=['chr', 'start', 'end', 'count'])
        
    os.remove(main_path)
    os.remove(intersect_path)
    os.remove(result_path)

    return drop_duplicates_and_sort(intersected_df)

def intersect_colocalization_bed_files(df: pd.DataFrame, intersect_df: pd.DataFrame, include_count: bool = False) -> pd.DataFrame:
    col_names = df.columns.tolist()
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as main_file, \
         tempfile.NamedTemporaryFile(delete=False, mode='w') as intersect_file, \
         tempfile.NamedTemporaryFile(delete=False, mode='w') as result_file:
        
        main_path = main_file.name
        intersect_path = intersect_file.name
        result_path = result_file.name

        df.to_csv(main_path, sep="\t", index=False, header=False)
        intersect_df.to_csv(intersect_path, sep="\t", index=False, header=False)

        command = f"bedtools intersect -a {main_path} -b {intersect_path} -wa -wb > {result_path}"
        run_bedtools_command(command)

        intersected_df = pd.read_csv(result_path, sep="\t", header=None)
        
        if include_count:
            intersected_df = intersected_df.iloc[:, [-4, -3, -2, -1] + [i + 3 for i in range(len(col_names) - 3)]]
            intersected_df.columns = ["chr", "start", "end", "count_2"] + col_names[3:]
        else:
            intersected_df = intersected_df.iloc[:, [-3, -2, -1] + [i + 3 for i in range(len(col_names) - 3)]]
            intersected_df.columns = col_names

    os.remove(main_path)
    os.remove(intersect_path)
    os.remove(result_path)

    return drop_duplicates_and_sort(intersected_df)

# ============================================================
# Data Processing Functions
# ============================================================
def assign_negative_label(df: pd.DataFrame) -> pd.DataFrame:
    df["label"] = NEGATIVE_LABEL
    df["count"] = 0
    return df

def balance_labels(df: pd.DataFrame) -> pd.DataFrame:
    min_count = df["label"].value_counts().min()
    # Explicitly select all columns to keep them in the result
    columns_to_keep = df.columns
    return df.groupby("label")[columns_to_keep].apply(lambda x: x.sample(min_count)).reset_index(drop=True)

def label_positive_peaks(df: pd.DataFrame) -> pd.DataFrame:
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    if DataFilterOption.FILTER_DATA:
        max_count = df["count"].max()
        if max_count <= 2:
            df["label"] = POSITIVE_LABEL
            return df
        elif max_count > MAX_COUNT_THRESHOLD:
            threshold = df["count"].quantile(HIGH_COUNT_QUANTILE)
            df = df[df["count"] > threshold]
        elif max_count > MID_COUNT_THRESHOLD:
            threshold = df["count"].median()
            df = df[df["count"] > threshold]
    
    df["label"] = POSITIVE_LABEL
    return df

# ============================================================
# Process ATAC Peaks and Label Them with Chip Data
# ============================================================
def process_atac_files(chip_input_dir: str, cell_lines: dict, args) -> list:
    processed_dfs = []
    # Get cell lines present in the merged directory
    present_cell_lines = {
        filename.split("_")[0]
        for filename in os.listdir(chip_input_dir)
        if filename.endswith(".bed")
    }
    # Only keep mappings for cell lines that have chip files and ATAC files
    filtered_cell_lines = {
        k: v for k, v in cell_lines.items() 
        if k in present_cell_lines and validate_file_exists(
            os.path.join(args.aligned_chip_data_dir, v, "peaks", f"{v}.filtered.broadPeak")
        )
    }

    for cell_line_key, cell_line in filtered_cell_lines.items():
        atac_path = os.path.join(
            args.aligned_chip_data_dir, cell_line, "peaks", f"{cell_line}.filtered.broadPeak"
        )
        atac_df = pd.read_csv(atac_path, sep="\t", header=None, names=["chr", "start", "end"], usecols=[0, 1, 2])
        atac_df = filter_by_peak_length(atac_df)
        atac_df = filter_by_chromosomes(atac_df)
        chip_files = []
        for filename in os.listdir(chip_input_dir):
            if not filename.endswith(".bed"):
                continue
            if filename.split("_")[0] == cell_line_key:
                chip_files.append(os.path.join(chip_input_dir, filename))

        if chip_files:
            chip_dfs = []
            for file in chip_files:
                chip_df = pd.read_csv(file, sep="\t", header=None, names=["chr", "start", "end", "count"])
                chip_df = filter_by_peak_length(chip_df)
                chip_df = filter_by_chromosomes(chip_df)
                chip_dfs.append(chip_df)
            if chip_dfs:
                chip_df_combined = pd.concat(chip_dfs, ignore_index=True)
                chip_df_combined = drop_duplicates_and_sort(chip_df_combined)
                atac_positive = intersect_bed_files(atac_df, chip_df_combined)
                atac_positive = label_positive_peaks(atac_positive)
                atac_negative = subtract_bed_files(atac_df, chip_df_combined)
                atac_negative = assign_negative_label(atac_negative)
                atac_positive["cell_line"] = cell_line
                atac_negative["cell_line"] = cell_line
                processed_dfs.append(atac_positive)
                processed_dfs.append(atac_negative)
            else:
                logging.warning(f"No valid chip peaks found for {cell_line_key}")
                atac_df = assign_negative_label(atac_df)
                atac_df["cell_line"] = cell_line
                processed_dfs.append(atac_df)
        else:
            logging.warning(f"No chip files found for cell line key {cell_line_key}")
            atac_df = assign_negative_label(atac_df)
            atac_df["cell_line"] = cell_line
            processed_dfs.append(atac_df)
    
    return processed_dfs

def split_dataset(combined_df: pd.DataFrame, args):
    if args.validation_cell_lines:
        validation_set = combined_df[combined_df["cell_line"].isin(args.validation_cell_lines)]
        training_set = combined_df[~combined_df["cell_line"].isin(args.validation_cell_lines)]
        logging.info(f"Validation cell lines: {args.validation_cell_lines}")
    elif args.validation_chromosomes:
        validation_set = combined_df[combined_df["chr"].isin(args.validation_chromosomes)]
        training_set = combined_df[~combined_df["chr"].isin(args.validation_chromosomes)]
        logging.info(f"Validation chromosomes: {args.validation_chromosomes}")
    else:
        validation_set = combined_df.sample(frac=DEFAULT_VALIDATION_SPLIT, random_state=42)
        training_set = combined_df.drop(validation_set.index)
        logging.info("No validation set specified, using default 20% split")
    return training_set, validation_set

def save_dataset(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, sep="\t", index=False)

# ============================================================
# Cell Line Mapping Check
# ============================================================
def check_cell_lines_in_chip(chip_input_dir: str, cell_lines: dict, args) -> None:
    chip_cell_lines = {
        filename.split("_")[0]
        for filename in os.listdir(chip_input_dir)
        if filename.endswith(".bed")
    }
    
    # Show all available ChIP files
    logging.info("\nAll available ChIP cell lines in directory:")
    for cell_line in sorted(chip_cell_lines):
        if cell_line in cell_lines:
            atac_path = os.path.join(
                args.aligned_chip_data_dir, cell_lines[cell_line], "peaks", 
                f"{cell_lines[cell_line]}.filtered.broadPeak"
            )
            if validate_file_exists(atac_path):
                status = "✓ Will be used"
            else:
                status = "✗ Skipped (ATAC file not found in provided aligned_chip_data_dir)"
        else:
            status = "✗ Skipped (not in cell_line_mapping.json)"
        logging.info(f"  - {cell_line}: {status}")
    
    usable_cell_lines = {
        cell_line for cell_line in chip_cell_lines & set(cell_lines.keys())
        if validate_file_exists(os.path.join(
            args.aligned_chip_data_dir, cell_lines[cell_line], "peaks", 
            f"{cell_lines[cell_line]}.filtered.broadPeak"
        ))
    }

    logging.info("\n" + "="*50)
    logging.info(f"Total ChIP files: {len(chip_cell_lines)}")
    logging.info(f"Will process: {len(usable_cell_lines)} cell lines")
    logging.info(f"Will skip: {len(chip_cell_lines) - len(usable_cell_lines)} cell lines")
    logging.info("\n" + "="*50)

# ============================================================
# Argument Parsing
# ============================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process TF binding data with ATAC primary input.")
    parser.add_argument("--tf", type=str, help="Transcription Factor")
    parser.add_argument("--balance", action="store_true", help="Balance dataset labels")
    parser.add_argument("--dont_filter", action="store_true", help="Do not filter data")
    parser.add_argument("--positive_only", action="store_true", help="Return only positive samples")
    validation_group = parser.add_mutually_exclusive_group(required=False)
    validation_group.add_argument("--validation_cell_lines", type=str, nargs="*", help="Cell lines for validation set")
    validation_group.add_argument("--validation_chromosomes", type=str, nargs="*", help="Chromosomes for validation set")
    
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--contrasting_tf", type=str, help="Transcription factor for negative dataset")
    group.add_argument("--colocalization_tf", type=str, help="Transcription factor for colocalization dataset")
    parser.add_argument("--chip_provided", action="store_true", help="Use provided ChIP data for negative set")
    parser.add_argument("--cell_line_mapping", type=str, default="cell_line_mapping.json")
    
    parser.add_argument("--tf_base_dir", type=str,
                        default="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/transcription_factors")
    parser.add_argument("--aligned_chip_data_dir", type=str,
                        default="/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines")
    
    parser.add_argument("--output_dir", type=str,
                        default="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/data_splits")
    parser.add_argument("--validation_file", type=str, default="validation_combined.csv")

    parser.add_argument("--no_ground_truth", action="store_true", help="Prepare data for prediction without ground truth")
    parser.add_argument("--input_bed_file", type=str, help="Input BED file path for prediction")
    parser.add_argument("--output_file", type=str, default="prediction_data.csv")
    parser.add_argument("--negative_regions_bed", type=str, nargs="+", help="Path(s) to BED file(s) for negative samples")

    args = parser.parse_args()

    if args.chip_provided and not args.colocalization_tf:
        parser.error("--chip_provided requires --colocalization_tf")
    if args.contrasting_tf:
        raise NotImplementedError("The feature 'contrasting_tf' is not implemented yet")
    if args.no_ground_truth and not args.input_bed_file:
        parser.error("--input_bed_file is required when --no_ground_truth is set")
    if not args.no_ground_truth and not args.tf:
        parser.error("--tf is required unless --no_ground_truth is set")
    return args

def load_cell_line_mapping(file_path: str) -> dict:
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"Error loading cell line mapping: {e}")
        return {}

# ============================================================
# Main
# ============================================================
def main():
    args = parse_arguments()

    # Convert arguments to enums
    args.balance_option = BalanceOption.BALANCE_LABELS if args.balance else BalanceOption.NO_BALANCE
    args.region_filter_option = RegionFilterOption.ALL_REGIONS
    args.data_filter_option = DataFilterOption.NO_FILTER if args.dont_filter else DataFilterOption.FILTER_DATA
    args.label_option = LabelOption.POSITIVE_ONLY if args.positive_only else LabelOption.BOTH_LABELS
    args.chip_data_option = ChipDataOption.CHIP_PROVIDED if args.chip_provided else ChipDataOption.NO_CHIP
    args.ground_truth_option = GroundTruthOption.NO_GROUND_TRUTH if args.no_ground_truth else GroundTruthOption.WITH_GROUND_TRUTH

    os.makedirs(args.output_dir, exist_ok=True)

    if args.ground_truth_option == GroundTruthOption.NO_GROUND_TRUTH:
        df = pd.read_csv(args.input_bed_file, sep="\t", header=None, names=["chr", "start", "end"], usecols=[0, 1, 2])
        df = filter_by_peak_length(df)
        df = filter_by_chromosomes(df)
        df["count"] = NO_GROUND_TRUTH_LABEL
        df["label"] = NO_GROUND_TRUTH_LABEL
        df["cell_line"] = os.path.basename(args.input_bed_file).split('.')[0]
        output_path = os.path.join(args.output_dir, args.output_file)
        save_dataset(df, output_path)
        logging.info(f"Prediction data saved to: {output_path}")
        return

    tf = args.tf
    # The chip peaks are still stored in the merged directory
    chip_input_dir = os.path.join(args.tf_base_dir, tf, "merged")
    cell_lines = load_cell_line_mapping(args.cell_line_mapping)

    # Check which chip files are available
    check_cell_lines_in_chip(chip_input_dir, cell_lines, args)

    # Process ATAC peaks and label them according to chip overlap
    processed = process_atac_files(chip_input_dir, cell_lines, args)
    if not processed:
        logging.error("No data processed. Check the ATAC directory and the cell line mapping file.")
        return

    combined_df = pd.concat(processed, ignore_index=True)

    if args.label_option == LabelOption.POSITIVE_ONLY:
        combined_df = combined_df[combined_df["label"] == POSITIVE_LABEL]

    if args.balance_option == BalanceOption.BALANCE_LABELS:
        # Explicitly select all columns to keep them in the result
        columns_to_keep = combined_df.columns
        combined_df = combined_df.groupby("cell_line")[columns_to_keep].apply(balance_labels).reset_index(drop=True)

    training_set, validation_set = split_dataset(combined_df, args)
    training_set_file = os.path.join(args.output_dir, "training_combined.csv")
    validation_set_file = os.path.join(args.output_dir, args.validation_file)

    save_dataset(training_set, training_set_file)
    save_dataset(validation_set, validation_set_file)

    # Extract cell lines used in training and validation
    training_cell_lines = training_set['cell_line'].unique().tolist()
    validation_cell_lines = validation_set['cell_line'].unique().tolist()
    
    # Save cell line info to a JSON file
    cell_line_info = {
        'training_cell_lines': training_cell_lines,
        'validation_cell_lines': validation_cell_lines
    }
    
    cell_line_info_file = os.path.join(args.output_dir, "cell_line_info.json")
    with open(cell_line_info_file, 'w') as f:
        json.dump(cell_line_info, f, indent=2)
    
    # Log the number of positive and negative hits for each cell line
    for cell_line, group in combined_df.groupby('cell_line'):
        positive_hits = group[group['label'] == POSITIVE_LABEL].shape[0]
        negative_hits = group[group['label'] == NEGATIVE_LABEL].shape[0]
        logging.info(f"Cell line {cell_line}: {positive_hits} positive hits, {negative_hits} negative hits")

    logging.info("\n" + "="*50)
    logging.info(f"Training set saved to: {training_set_file}")
    logging.info(f"Validation set saved to: {validation_set_file}")
    logging.info(f"Cell line information saved to: {cell_line_info_file}")
    logging.info(f"Training cell lines: {training_cell_lines}")
    logging.info(f"Validation cell lines: {validation_cell_lines}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()