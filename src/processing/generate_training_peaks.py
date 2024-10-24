import os
import sys
import json
import logging
import argparse

import pandas as pd
import numpy as np

from utils.bedtools import (
    subtract_bed_files,
    intersect_bed_files,
    intersect_colocalization_bed_files,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

def parse_arguments():
    """
    Parse command-line arguments using argparse.
    """
    parser = argparse.ArgumentParser(description="Process TF binding data.")

    parser.add_argument("tf", type=str, help="Transcription Factor")
    parser.add_argument(
        "--balance", action="store_true", help="Balance the labels in the datasets"
    )
    parser.add_argument(
        "--enhancer_promotor_only",
        action="store_true",
        help="Only consider enhancer and promoter regions",
    )
    parser.add_argument(
        "--dont_filter", action="store_true", help="Do not filter the data"
    )
    parser.add_argument(
        "--positive_only",
        action="store_true",
        help="Return only the peaks where the TF is bound",
    )

    # Mutually exclusive group for validation set
    validation_group = parser.add_mutually_exclusive_group(required=False)
    validation_group.add_argument(
        "--validation_cell_lines",
        type=str,
        nargs="*",
        help="Cell lines for validation set",
    )
    validation_group.add_argument(
        "--validation_chromosomes",
        type=str,
        nargs="*",
        help="Chromosomes for validation set",
    )

    # Mutually exclusive group for negative or colocalization TF
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--negative_tf",
        type=str,
        help="Transcription factor for negative dataset",
    )
    group.add_argument(
        "--colocalization_tf",
        type=str,
        help="Transcription factor for colocalization dataset",
    )

    parser.add_argument(
        "--chip_provided",
        action="store_true",
        help="Use the provided ChIP data for the negative set",
    )

    # New arguments for configurable paths
    parser.add_argument(
        "--cell_line_mapping",
        type=str,
        default="cell_line_mapping.json",
        help="Path to cell line mapping JSON file",
    )
    parser.add_argument(
        "--tf_base_dir",
        type=str,
        default="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors",
        help="Base directory for transcription factors data",
    )
    parser.add_argument(
        "--aligned_chip_data_dir",
        type=str,
        default="/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines",
        help="Base directory for aligned ChIP data",
    )
    parser.add_argument(
        "--enhancer_bed",
        type=str,
        default="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/GRCh38-ELS.bed",
        help="Path to enhancer BED file",
    )
    parser.add_argument(
        "--promoter_bed",
        type=str,
        default="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/GRCh38-PLS.bed",
        help="Path to promoter BED file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/data_splits",
        help="Output directory for the entire set",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.chip_provided and not args.colocalization_tf:
        parser.error("--chip_provided requires --colocalization_tf to be provided")

    if args.negative_tf:
        raise NotImplementedError("The feature 'negative_tf' is not implemented yet")

    return args

def load_cell_line_mapping(file_path):
    """
    Load cell line mappings from a JSON file.
    """
    try:
        with open(file_path, "r") as file:
            cell_lines = json.load(file)
        return cell_lines
    except Exception as e:
        logging.error(f"Error loading cell line mapping: {e}")
        return {}

def filter_peak_lengths(df, threshold=4000):
    """
    Filter peaks based on length threshold.
    """
    try:
        df = df[df["end"] - df["start"] <= threshold]
        return df
    except Exception as e:
        logging.error(f"Error filtering peak lengths: {e}")
        return pd.DataFrame()

def filter_chromosomes(df):
    """
    Keep only valid human chromosomes.
    """
    try:
        valid_chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
        df = df[df["chr"].isin(valid_chromosomes)]
        return df
    except Exception as e:
        logging.error(f"Error filtering chromosomes: {e}")
        return pd.DataFrame()

def label_peaks(df, balance=False, drop_rows=True):
    """
    Label peaks based on count and optionally balance the dataset.
    """
    try:
        df["label"] = 1
        if drop_rows:
            max_count = df["count"].max()
            if max_count <= 2:
                pass
            elif max_count > 30:
                threshold = df["count"].quantile(0.75)
                df = df[df["count"] > threshold]
            elif max_count > 10:
                threshold = df["count"].median()
                df = df[df["count"] > threshold]
            else:
                threshold = df["count"].median()
                df = df[df["count"] >= threshold]
        else:
            median_count = df["count"].median()
            df.loc[df["count"] < median_count, "label"] = 0
        return df
    except Exception as e:
        logging.error(f"Error labeling peaks: {e}")
        return pd.DataFrame()

def balance_labels(df):
    """
    Balance the dataset to have equal numbers of positive and negative samples.
    """
    try:
        min_count = df["label"].value_counts().min()
        balanced_df = df.groupby(["label"])[['chr', 'start', 'end', 'count', 'label', 'cell_line']].apply(lambda x: x.sample(min_count)).reset_index(drop=True)
        return balanced_df
    except Exception as e:
        logging.error(f"Error balancing labels: {e}")
        return df

def process_bed_files(input_dir, cell_lines, args):
    """
    Process BED files to create a combined dataset.
    """
    filtered_dfs = []

    for filename in os.listdir(input_dir):
        if not filename.endswith(".bed"):
            continue

        # Construct file paths
        input_filepath = os.path.join(input_dir, filename)
        cell_line_key = filename.split("_")[0]
        cell_line = cell_lines.get(cell_line_key)

        if not cell_line:
            logging.warning(f"Cell line not found for key {cell_line_key}")
            continue

        # Read input BED file
        df = pd.read_csv(
            input_filepath,
            sep="\t",
            header=None,
            names=["chr", "start", "end", "count"],
        )

        # Filter peaks and chromosomes
        df = filter_peak_lengths(df)
        df = filter_chromosomes(df)

        # Label peaks
        df = label_peaks(df, balance=args.balance, drop_rows=not args.dont_filter)
        df["cell_line"] = cell_line

        # Process colocalization if specified
        if args.colocalization_tf:
            colocalization_data = process_colocalization(
                df, cell_line_key, cell_line, args
            )
            if colocalization_data:
                filtered_dfs.extend(colocalization_data)
        else:
            negative_df = process_negative_samples(df, cell_line, args)
            filtered_dfs.append(df)
            filtered_dfs.append(negative_df)

    return filtered_dfs

def process_colocalization(df, cell_line_key, cell_line, args):
    """
    Process colocalization data with another transcription factor.
    """
    colocalization_filepath = os.path.join(
        args.tf_base_dir,
        args.colocalization_tf,
        "output",
        f"{cell_line_key}_{args.colocalization_tf}.bed"
    )

    if not os.path.exists(colocalization_filepath):
        logging.warning(
            f"Colocalization file not found for cell line {cell_line} at {colocalization_filepath}"
        )
        return None

    # Read colocalization BED file
    colocalization_df = pd.read_csv(
        colocalization_filepath,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "count"],
    )
    colocalization_df = filter_peak_lengths(colocalization_df)
    colocalization_df = filter_chromosomes(colocalization_df)

    # Process overlapping and negative samples
    overlapping_df, negative_df = get_overlapping_and_negative_samples(
        df, colocalization_df, cell_line, args
    )

    logging.info(
        f"Cell line: {cell_line}; Overlapping samples: {len(overlapping_df)}, Negative samples: {len(negative_df)}"
    )

    return [overlapping_df, negative_df]

def get_overlapping_and_negative_samples(df1, df2, cell_line, args):
    """
    Get overlapping and negative samples between two DataFrames.
    """
    # Intersect with ATAC-seq data
    atac_filepath = os.path.join(
        args.aligned_chip_data_dir,
        cell_line,
        "peaks",
        f"{cell_line}.filtered.broadPeak"
    )

    atac_df = pd.read_csv(
        atac_filepath,
        sep="\t",
        usecols=[0, 1, 2],
        header=None,
        names=["chr", "start", "end"],
    )
    atac_df = filter_chromosomes(atac_df)

    # Intersect with ATAC-seq peaks
    tf1_df = intersect_colocalization_bed_files(df1, atac_df)
    tf2_df = intersect_colocalization_bed_files(df2, atac_df)

    # Get overlapping peaks
    overlapping_df = intersect_colocalization_bed_files(
        tf1_df, tf2_df[["chr", "start", "end", "count"]], count_included=True
    )
    overlapping_df["cell_line"] = cell_line
    overlapping_df["label"] = 1

    # Get negative samples
    if args.chip_provided:
        negative_df = subtract_bed_files(tf1_df, tf2_df[["chr", "start", "end"]])
    else:
        negative_df = subtract_bed_files(atac_df, tf1_df[["chr", "start", "end"]])
        negative_df = subtract_bed_files(negative_df, tf2_df[["chr", "start", "end"]])

    negative_df["cell_line"] = cell_line
    negative_df["label"] = 0
    negative_df["count"] = 0
    negative_df["count_2"] = 0

    return overlapping_df, negative_df

def process_negative_samples(df, cell_line, args):
    """
    Process negative samples for the given cell line.
    """
    negative_filepath = os.path.join(
        args.aligned_chip_data_dir,
        cell_line,
        "peaks",
        f"{cell_line}.filtered.broadPeak"
    )

    if not os.path.exists(negative_filepath):
        logging.warning(
            f"Negative samples file not found for cell line {cell_line} at {negative_filepath}"
        )
        return pd.DataFrame()

    # Read negative samples
    neg_df = pd.read_csv(
        negative_filepath,
        sep="\t",
        usecols=[0, 1, 2],
        header=None,
        names=["chr", "start", "end"],
    )
    neg_df = filter_peak_lengths(neg_df)
    neg_df = filter_chromosomes(neg_df)

    # Subtract positive samples
    neg_df = subtract_bed_files(neg_df, df[["chr", "start", "end"]])
    neg_df["cell_line"] = cell_line
    neg_df["label"] = 0
    neg_df["count"] = 0

    # TODO: If sum(df['label'] == 1) == 0, then don't add to filtered_dfs

    logging.info(
        f"Cell line: {cell_line}; Positive samples: {df.shape[0]}, Negative samples: {neg_df.shape[0]}"
    )

    return neg_df

def process_enhancer_promoter_regions(combined_df, args):
    """
    Filter the combined DataFrame for enhancer and promoter regions.
    """
    enhancer_df = pd.read_csv(
        args.enhancer_bed,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "EH38D", "EH38E", "feature_type"],
    )

    promoter_df = pd.read_csv(
        args.promoter_bed,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "EL38D", "EL38E", "feature_type"],
    )

    # Intersect with enhancer and promoter regions
    enhancer_intersect = intersect_bed_files(combined_df, enhancer_df, "enhancer")
    promoter_intersect = intersect_bed_files(combined_df, promoter_df, "promoter")

    # Combine and one-hot encode region types
    combined_filtered_df = pd.concat([enhancer_intersect, promoter_intersect])

    # TODO: Could add in test set option, as well as the option to just sample for the validation set
    # TODO: Could oversample minority class instead of undersampling majority class to keep data

    combined_filtered_df = pd.get_dummies(combined_filtered_df, columns=["region_type"])

    return combined_filtered_df

def split_dataset(combined_df, args):
    """
    Split the combined DataFrame into training and validation sets.
    """
    if args.validation_cell_lines:
        validation_set = combined_df[combined_df["cell_line"].isin(args.validation_cell_lines)]
        training_set = combined_df[~combined_df["cell_line"].isin(args.validation_cell_lines)]
        logging.info(f"Validation cell lines: {args.validation_cell_lines}")
    elif args.validation_chromosomes:
        validation_set = combined_df[combined_df["chr"].isin(args.validation_chromosomes)]
        training_set = combined_df[~combined_df["chr"].isin(args.validation_chromosomes)]
        logging.info(f"Validation chromosomes: {args.validation_chromosomes}")
    else:
        validation_set = combined_df.sample(frac=0.2, random_state=42)
        training_set = combined_df.drop(validation_set.index)
        logging.info("No validation set specified, using default 20% split")

    return training_set, validation_set

def save_datasets(training_set, validation_set, output_dir):
    """
    Save the training and validation sets to CSV files.
    """
    training_set_file = os.path.join(output_dir, "training_combined.csv")
    validation_set_file = os.path.join(output_dir, "validation_combined.csv")

    training_set.to_csv(training_set_file, sep="\t", index=False)
    validation_set.to_csv(validation_set_file, sep="\t", index=False)

    logging.info(f"Training set saved to: {training_set_file}")
    logging.info(f"Validation set saved to: {validation_set_file}")

def main():
    """
    Main function to orchestrate the processing of TF binding data.
    """
    args = parse_arguments()

    # Set up paths
    tf = args.tf
    input_dir = os.path.join(args.tf_base_dir, tf, "output")

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load cell line mapping
    cell_lines = load_cell_line_mapping(args.cell_line_mapping)

    # Process BED files
    filtered_dfs = process_bed_files(input_dir, cell_lines, args)
    if not filtered_dfs:
        logging.error("No data processed. Exiting.")
        return

    # Combine DataFrames
    combined_df = pd.concat(filtered_dfs, ignore_index=True)

    # Process enhancer and promoter regions if specified
    # if args.enhancer_promotor_only:
    #     combined_df = process_enhancer_promoter_regions(combined_df, args)
    # else:
    #     combined_df["region_type_enhancer"] = np.nan
    #     combined_df["region_type_promoter"] = np.nan

    # Keep only positive samples if specified
    if args.positive_only:
        combined_df = combined_df[combined_df["label"] == 1]

    # Balance labels if specified
    if args.balance:
        combined_df = combined_df.groupby(["cell_line"])[["chr", "start", "end", "count", "label", "cell_line"]].apply(balance_labels).reset_index(drop=True)

    # Split dataset into training and validation sets
    training_set, validation_set = split_dataset(combined_df, args)

    # Save datasets
    save_datasets(training_set, validation_set, args.output_dir)

if __name__ == "__main__":
    # set wd to where this file is
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
