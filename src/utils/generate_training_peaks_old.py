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
    ENHANCER_PROMOTER_ONLY = "enhancer_promoter_only"
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

def intersect_bed_files(main_df: pd.DataFrame, intersect_df: pd.DataFrame, region_type: str = None) -> pd.DataFrame:
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

        intersected_df = pd.read_csv(result_path, sep="\t", header=None, usecols=range(len(col_names)), names=col_names)
        
    os.remove(main_path)
    os.remove(intersect_path)
    os.remove(result_path)

    intersected_df = drop_duplicates_and_sort(intersected_df)

    if region_type:
        intersected_df["region_type"] = region_type
    return intersected_df

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
def label_peaks(peaks_df: pd.DataFrame, balance_option: BalanceOption, data_filter_option: DataFilterOption) -> pd.DataFrame:
    peaks_df["label"] = POSITIVE_LABEL
    if data_filter_option == DataFilterOption.NO_FILTER:
        # Apply filtering logic
        max_count = peaks_df["count"].max()
        if max_count <= 2:
            return peaks_df
        elif max_count > MAX_COUNT_THRESHOLD:
            threshold = peaks_df["count"].quantile(HIGH_COUNT_QUANTILE)
            peaks_df = peaks_df[peaks_df["count"] > threshold]
        elif max_count > MID_COUNT_THRESHOLD:
            threshold = peaks_df["count"].median()
            peaks_df = peaks_df[peaks_df["count"] > threshold]
        else:
            threshold = peaks_df["count"].median()
            peaks_df = peaks_df[peaks_df["count"] >= threshold]
    return peaks_df

def assign_negative_label(df: pd.DataFrame) -> pd.DataFrame:
    df["label"] = NEGATIVE_LABEL
    df["count"] = 0
    return df

def balance_labels(df: pd.DataFrame) -> pd.DataFrame:
    min_count = df["label"].value_counts().min()
    return df.groupby("label").apply(lambda x: x.sample(min_count)).reset_index(drop=True)

# ============================================================
# Specialized Processing
# ============================================================
def get_negative_samples(df: pd.DataFrame, cell_line: str, aligned_chip_dir: str, negative_regions_files: list) -> pd.DataFrame:
    neg_path = os.path.join(aligned_chip_dir, cell_line, "peaks", f"{cell_line}.filtered.broadPeak")
    if not validate_file_exists(neg_path):
        logging.warning(f"Negative samples file not found for {cell_line} at {neg_path}")
        return pd.DataFrame()

    neg_df = pd.read_csv(neg_path, sep="\t", header=None, names=["chr", "start", "end"], usecols=[0, 1, 2])
    neg_df = filter_by_peak_length(neg_df)
    neg_df = filter_by_chromosomes(neg_df)
    neg_df = subtract_bed_files(neg_df, df[["chr", "start", "end"]])

    if negative_regions_files:
        neg_df_overlapping = pd.DataFrame()
        for bed_file in negative_regions_files:
            if not validate_file_exists(bed_file):
                logging.warning(f"Negative regions file not found: {bed_file}")
                continue
            neg_regions_df = pd.read_csv(bed_file, sep="\t", header=None, names=["chr", "start", "end"])
            neg_regions_df = filter_by_chromosomes(neg_regions_df)
            current_overlap = intersect_bed_files(neg_df, neg_regions_df)
            neg_df_overlapping = pd.concat([neg_df_overlapping, current_overlap], ignore_index=True)

        neg_df_overlapping = neg_df_overlapping.drop_duplicates(subset=["chr", "start", "end"])
        num_positives = df.shape[0]
        num_negatives_current = neg_df_overlapping.shape[0]
        if num_negatives_current < num_positives:
            remaining_neg_df = neg_df[~neg_df.apply(tuple, axis=1).isin(neg_df_overlapping.apply(tuple, axis=1))]
            additional_negatives = remaining_neg_df.sample(
                n=min(num_positives - num_negatives_current, len(remaining_neg_df)),
                replace=False
            )
            additional_negatives = assign_negative_label(additional_negatives)
            additional_negatives["cell_line"] = cell_line
            neg_df_final = pd.concat([neg_df_overlapping, additional_negatives], ignore_index=True)
        else:
            neg_df_final = neg_df_overlapping.sample(n=num_positives, replace=False)
    else:
        neg_df_final = neg_df

    neg_df_final = assign_negative_label(neg_df_final)
    neg_df_final["cell_line"] = cell_line
    logging.info(f"Cell line: {cell_line}; Positive: {df.shape[0]}, Negative: {neg_df_final.shape[0]}")
    return neg_df_final

def process_colocalization(df: pd.DataFrame, cell_line_key: str, cell_line: str, args) -> list:
    colocalization_path = os.path.join(
        args.tf_base_dir,
        args.colocalization_tf,
        "output",
        f"{cell_line_key}_{args.colocalization_tf}.bed"
    )
    if not validate_file_exists(colocalization_path):
        logging.warning(f"Colocalization file not found for {cell_line} at {colocalization_path}")
        return []

    colocal_df = pd.read_csv(colocalization_path, sep="\t", header=None, names=["chr", "start", "end", "count"])
    colocal_df = filter_by_peak_length(colocal_df)
    colocal_df = filter_by_chromosomes(colocal_df)

    overlapping_df, negative_df = get_overlapping_and_negative_samples(df, colocal_df, cell_line, args)
    return [overlapping_df, negative_df]

def get_overlapping_and_negative_samples(df1: pd.DataFrame, df2: pd.DataFrame, cell_line: str, args) -> (pd.DataFrame, pd.DataFrame):
    atac_path = os.path.join(args.aligned_chip_data_dir, cell_line, "peaks", f"{cell_line}.filtered.broadPeak")
    atac_df = pd.read_csv(atac_path, sep="\t", usecols=[0, 1, 2], header=None, names=["chr", "start", "end"])
    atac_df = filter_by_chromosomes(atac_df)

    tf1_df = intersect_colocalization_bed_files(df1, atac_df, include_count=False)
    tf2_df = intersect_colocalization_bed_files(df2, atac_df, include_count=False)
    overlapping_df = intersect_colocalization_bed_files(tf1_df, tf2_df[["chr", "start", "end", "count"]], include_count=True)
    overlapping_df["cell_line"] = cell_line
    overlapping_df["label"] = POSITIVE_LABEL

    if args.chip_data_option == ChipDataOption.CHIP_PROVIDED:
        negative_df = subtract_bed_files(tf1_df, tf2_df[["chr", "start", "end"]])
    else:
        negative_df = subtract_bed_files(atac_df, tf1_df[["chr", "start", "end"]])
        negative_df = subtract_bed_files(negative_df, tf2_df[["chr", "start", "end"]])

    negative_df = assign_negative_label(negative_df)
    negative_df["count_2"] = 0
    negative_df["cell_line"] = cell_line

    logging.info(f"Cell line: {cell_line}; Overlapping: {len(overlapping_df)}, Negative: {len(negative_df)}")
    return overlapping_df, negative_df

def process_bed_files(input_dir: str, cell_lines: dict, args) -> list:
    processed_dfs = []
    for filename in os.listdir(input_dir):
        if not filename.endswith(".bed"):
            continue

        cell_line_key = filename.split("_")[0]
        cell_line = cell_lines.get(cell_line_key)
        if not cell_line:
            logging.warning(f"Cell line not found for key {cell_line_key}")
            continue

        input_path = os.path.join(input_dir, filename)
        df = pd.read_csv(input_path, sep="\t", header=None, names=["chr", "start", "end", "count"])
        df = filter_by_peak_length(df)
        df = filter_by_chromosomes(df)
        df = label_peaks(df, args.balance_option, args.data_filter_option)
        df["cell_line"] = cell_line

        if args.colocalization_tf:
            colocal_data = process_colocalization(df, cell_line_key, cell_line, args)
            processed_dfs.extend(colocal_data)
        else:
            neg_df = get_negative_samples(df, cell_line, args.aligned_chip_data_dir, args.negative_regions_bed)
            processed_dfs.append(df)
            processed_dfs.append(neg_df)

    return processed_dfs

def process_enhancer_promoter_regions(combined_df: pd.DataFrame, enhancer_bed: str, promoter_bed: str) -> pd.DataFrame:
    enhancer_df = pd.read_csv(enhancer_bed, sep="\t", header=None,
                              names=["chr", "start", "end", "EH38D", "EH38E", "feature_type"])
    promoter_df = pd.read_csv(promoter_bed, sep="\t", header=None,
                              names=["chr", "start", "end", "EL38D", "EL38E", "feature_type"])

    enhancer_intersect = intersect_bed_files(combined_df, enhancer_df, "enhancer")
    promoter_intersect = intersect_bed_files(combined_df, promoter_df, "promoter")

    combined_filtered_df = pd.concat([enhancer_intersect, promoter_intersect])
    combined_filtered_df = pd.get_dummies(combined_filtered_df, columns=["region_type"])
    return combined_filtered_df

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
# Argument Parsing
# ============================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process TF binding data.")
    parser.add_argument("--tf", type=str, help="Transcription Factor")
    parser.add_argument("--balance", action="store_true", help="Balance dataset labels")
    parser.add_argument("--enhancer_promotor_only", action="store_true", help="Consider only enhancer/promoter regions")
    parser.add_argument("--dont_filter", action="store_true", help="Do not filter data")
    parser.add_argument("--positive_only", action="store_true", help="Return only positive samples")
    validation_group = parser.add_mutually_exclusive_group(required=False)
    validation_group.add_argument("--validation_cell_lines", type=str, nargs="*", help="Cell lines for validation set")
    validation_group.add_argument("--validation_chromosomes", type=str, nargs="*", help="Chromosomes for validation set")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--negative_tf", type=str, help="Transcription factor for negative dataset")
    group.add_argument("--colocalization_tf", type=str, help="Transcription factor for colocalization dataset")
    parser.add_argument("--chip_provided", action="store_true", help="Use provided ChIP data for negative set")
    parser.add_argument("--cell_line_mapping", type=str, default="cell_line_mapping.json")
    parser.add_argument("--tf_base_dir", type=str,
                        default="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/transcription_factors")
    parser.add_argument("--aligned_chip_data_dir", type=str,
                        default="/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines")
    parser.add_argument("--enhancer_bed", type=str,
                        default="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/GRCh38-ELS.bed")
    parser.add_argument("--promoter_bed", type=str,
                        default="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/GRCh38-PLS.bed")
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

    if args.negative_tf:
        raise NotImplementedError("The feature 'negative_tf' is not implemented yet")

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
    args.region_filter_option = RegionFilterOption.ENHANCER_PROMOTER_ONLY if args.enhancer_promotor_only else RegionFilterOption.ALL_REGIONS
    args.data_filter_option = DataFilterOption.NO_FILTER if args.dont_filter else DataFilterOption.FILTER_DATA
    args.label_option = LabelOption.POSITIVE_ONLY if args.positive_only else LabelOption.BOTH_LABELS
    args.chip_data_option = ChipDataOption.CHIP_PROVIDED if args.chip_provided else ChipDataOption.NO_CHIP
    args.ground_truth_option = GroundTruthOption.NO_GROUND_TRUTH if args.no_ground_truth else GroundTruthOption.WITH_GROUND_TRUTH

    os.makedirs(args.output_dir, exist_ok=True)

    if args.ground_truth_option == GroundTruthOption.NO_GROUND_TRUTH:
        # Prediction mode
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

    # Training/Validation mode
    tf = args.tf
    input_dir = os.path.join(args.tf_base_dir, tf, "output")
    cell_lines = load_cell_line_mapping(args.cell_line_mapping)

    processed = process_bed_files(input_dir, cell_lines, args)
    if not processed:
        logging.error("No data processed. Check the input directory and the cell line mapping file.")
        return

    combined_df = pd.concat(processed, ignore_index=True)

    if args.region_filter_option == RegionFilterOption.ENHANCER_PROMOTER_ONLY:
        combined_df = process_enhancer_promoter_regions(combined_df, args.enhancer_bed, args.promoter_bed)

    if args.label_option == LabelOption.POSITIVE_ONLY:
        combined_df = combined_df[combined_df["label"] == POSITIVE_LABEL]

    if args.balance_option == BalanceOption.BALANCE_LABELS:
        combined_df = combined_df.groupby("cell_line").apply(balance_labels).reset_index(drop=True)

    training_set, validation_set = split_dataset(combined_df, args)
    training_set_file = os.path.join(args.output_dir, "training_combined.csv")
    validation_set_file = os.path.join(args.output_dir, args.validation_file)

    save_dataset(training_set, training_set_file)
    save_dataset(validation_set, validation_set_file)

    logging.info(f"Training set saved to: {training_set_file}")
    logging.info(f"Validation set saved to: {validation_set_file}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
