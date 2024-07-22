import pandas as pd
import os
import numpy as np
import argparse
import json
import logging
from utils.bedtools import subtract_bed_files, intersect_bed_files

# Set up argument parser
parser = argparse.ArgumentParser(description="Process TF binding data.")
group = parser.add_mutually_exclusive_group(required=False)

parser.add_argument("tf", type=str, help="Transcription Factor")
parser.add_argument("--validation_cell_lines", type=str, nargs='*', help="Cell lines for validation set")
parser.add_argument('--balance', action='store_true', help='Balance the labels in the datasets')
parser.add_argument('--enhancer_promotor_only', action='store_true', help='Only consider enhancer and promotor regions')
parser.add_argument('--dont_filter', action='store_true', help='Dont filter the data')

group.add_argument('--negative_tf', type=str, help='Transcription factor for negative dataset')
group.add_argument('--colocalization_tf', type=str, help='Transcription factor for colocalization dataset')

args = parser.parse_args()

tf = args.tf
validation_cell_lines = args.validation_cell_lines if args.validation_cell_lines else None
balance = args.balance
enhancer_promotor_only = args.enhancer_promotor_only
negative_tf = args.negative_tf if args.negative_tf else None
colocalization_tf = args.colocalization_tf if args.colocalization_tf else None
dont_filter = args.dont_filter

input_dir = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{tf}/output"
output_file = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{tf}/entire_set/output_{tf}.csv"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def filter_df(df, threshold=4000):
    try:
        df = df[df['end'] - df['start'] <= threshold]
        if df['count'].max() > 50:
            return df.query('count >= @df["count"].quantile(.75)')
        elif df['count'].max() <= 2:
            return df
        else:
            return df.query('count > @df["count"].median()')
    except Exception as e:
        logging.error(f"Error filtering DataFrame: {e}")
        return pd.DataFrame()

def filter_chromosomes(df, threshold=4000):
    try:
        df = df[df['end'] - df['start'] <= threshold]
        valid_chromosomes = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY']
        return df[df['chr'].isin(valid_chromosomes)]
    except Exception as e:
        logging.error(f"Error filtering chromosomes: {e}")
        return pd.DataFrame()

def balance_labels(df):
    try:
        labels = df['label'].unique()
        
        min_count = min(df['label'].value_counts())
        
        balanced_df = pd.concat([df[df['label'] == label].sample(min_count) for label in labels])
        
        return balanced_df
    except Exception as e:
        logging.error(f"Error balancing labels: {e}")
        return df

def load_cell_line_mapping(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"Error loading cell line mapping: {e}")
        return {}

def process_bed_files(input_dir, cell_lines):
    filtered_dfs = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".bed"):
            input_filepath = os.path.join(input_dir, filename)
            df = pd.read_csv(input_filepath, sep="\t", header=None, names=['chr', 'start', 'end', 'count'])
            if dont_filter:
                filtered_df = df
            else:
                filtered_df = filter_df(df)
            filtered_df = filter_chromosomes(filtered_df).copy()
            cell_line_key = filename.split('_')[0]
            cell_line = cell_lines.get(cell_line_key)
            neg_df = pd.DataFrame()

            if cell_line:
                filtered_df['cell_line'] = cell_line
                filtered_df['label'] = 1
                if colocalization_tf:
                    colocalization_filepath = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{colocalization_tf}/output/{cell_line_key}_{colocalization_tf}.bed"
                    if os.path.exists(colocalization_filepath):
                        colocalization_df = pd.read_csv(colocalization_filepath, sep="\t", usecols=[0, 1, 2, 3], header=None, names=['chr', 'start', 'end', 'count'])
                        # logging.info(f"{tf} peaks: {len(filtered_df)}, {colocalization_tf} peaks: {len(colocalization_df)}")
                        overlapping_df = intersect_bed_files(filtered_df, colocalization_df[['chr', 'start', 'end']])
                        overlapping_df = filter_chromosomes(overlapping_df).copy()

                        neg_df = subtract_bed_files(filtered_df, colocalization_df[['chr', 'start', 'end']])
                        neg_df = filter_chromosomes(neg_df).copy()

                        colo_only_df = subtract_bed_files(colocalization_df, filtered_df[['chr', 'start', 'end']])
                        colo_only_df = filter_chromosomes(colo_only_df).copy()


                        overlapping_df['cell_line'] = cell_line
                        overlapping_df['label'] = 1

                        neg_df['cell_line'] = cell_line
                        neg_df['label'] = 0

                        filtered_dfs.append(overlapping_df)
                        filtered_dfs.append(neg_df)

                        logging.info(f"Colocalization samples between {tf} and {colocalization_tf} for {cell_line}: {len(overlapping_df)} overlapping samples, {len(neg_df)} {tf} only samples, {len(colo_only_df)} {colocalization_tf} only samples (not included in generated dataset)")
                    else:
                        logging.warning(f"Colocalization samples file not found for cell line {cell_line} at {colocalization_filepath}")

                elif negative_tf:
                    negative_filepath = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{negative_tf}/output/{cell_line_key}_{negative_tf}.bed"
                    if os.path.exists(negative_filepath):
                        contrasting_df = pd.read_csv(negative_filepath, sep="\t", usecols=[0, 1, 2, 3], header=None, names=['chr', 'start', 'end', 'count'])


                        neg_df = subtract_bed_files(contrasting_df, filtered_df[['chr', 'start', 'end']])

                        # Need to think about which way we want to do this (contrasting - filtered or filtered - contrasting)
                        overlapping_df = intersect_bed_files(contrasting_df, filtered_df[['chr', 'start', 'end']])

                        # subtract the positive samples from the negative samples
                        filtered_df = subtract_bed_files(filtered_df, contrasting_df[['chr', 'start', 'end']])

                        if dont_filter:
                            neg_df = neg_df
                        else:
                            neg_df = filter_df(neg_df)
                            overlapping_df = filter_df(overlapping_df)

                        neg_df = filter_chromosomes(neg_df).copy()
                        neg_df['cell_line'] = cell_line

                        overlapping_df = filter_chromosomes(overlapping_df).copy()
                        overlapping_df['cell_line'] = cell_line
                        filtered_df['label'] = f'{tf}'
                        neg_df['label'] = f'{negative_tf}'
                        overlapping_df['label'] = f'{tf}_{negative_tf}'

                        filtered_dfs.append(overlapping_df)
                        filtered_dfs.append(filtered_df)
                        filtered_dfs.append(neg_df)

                        logging.info(f"{filename}: {len(filtered_df)} positive samples, {len(neg_df)} negative samples, {len(overlapping_df)} overlapping samples")
                    else:
                        logging.warning(f"Negative samples file not found for cell line {cell_line} at {negative_filepath}")
                else:
                    negative_filepath = f"/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/{cell_line}/peaks/{cell_line}.filtered.broadPeak"
                    if os.path.exists(negative_filepath):
                        neg_df = pd.read_csv(negative_filepath, sep="\t", usecols=[0, 1, 2], header=None, names=['chr', 'start', 'end'])
                        neg_df = subtract_bed_files(neg_df, filtered_df[['chr', 'start', 'end']])
                        neg_df = filter_chromosomes(neg_df).copy()
                        neg_df['count'] = 1
                        neg_df['cell_line'] = cell_line
                        neg_df['label'] = 0
                        filtered_dfs.append(filtered_df)
                        filtered_dfs.append(neg_df)
                        logging.info(f"{filename}: {len(filtered_df)} positive samples, {len(neg_df)} negative samples")
                    else:
                        logging.warning(f"Negative samples file not found for cell line {cell_line} at {negative_filepath}")

            else:
                logging.warning(f"Cell line not found in mapping for key {cell_line_key}")

    return filtered_dfs


######################################################
######################## Main ########################
######################################################

cell_lines = load_cell_line_mapping('cell_line_mapping.json')
filtered_dfs = process_bed_files(input_dir, cell_lines)

if enhancer_promotor_only:
    combined_df = pd.concat(filtered_dfs)
    combined_df = combined_df[['chr', 'start', 'end', 'cell_line', 'label', 'count']]
    enhancer_df = pd.read_csv("/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/GRCh38-ELS.bed", sep="\t", header=None, names=['chr', 'start', 'end', 'EH38D','EH38E','feature_type'])
    promotor_df = pd.read_csv("/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/GRCh38-PLS.bed", sep="\t", header=None, names=['chr', 'start', 'end', 'EL38D','EL38E','feature_type'])
    combined_enhancer_df = intersect_bed_files(combined_df, enhancer_df, 'enhancer')
    combined_promotor_df = intersect_bed_files(combined_df, promotor_df, 'promoter')
    combined_filtered_df = pd.concat([combined_enhancer_df, combined_promotor_df])
    one_hot_encoded_df = pd.get_dummies(combined_filtered_df, columns=['region_type'])
else:
    combined_df = pd.concat(filtered_dfs)
    combined_df['region_type_enhancer'] = np.nan
    combined_df['region_type_promoter'] = np.nan
    one_hot_encoded_df = combined_df

grouped_df = one_hot_encoded_df.groupby(['chr', 'start', 'end', 'cell_line', 'label', 'count']).sum().reset_index()
os.makedirs(os.path.dirname(output_file), exist_ok=True)
grouped_df.to_csv(output_file, sep='\t', index=False)

# TODO: Could add in test set option, as well as the option to just sample for the validation set
# Balance the labels first
# TODO: Could oversample minority class instead of undersampling majority class to keep data
if balance:
    if negative_tf:
        grouped_df = grouped_df.groupby('cell_line').apply(balance_labels, include_groups=False).reset_index(drop=False).drop(columns="level_1")
    else:
        grouped_df = grouped_df.groupby('cell_line').apply(balance_labels, include_groups=False).reset_index(drop=False).drop(columns="level_1")

# Split the dataset into training and validation sets
if validation_cell_lines:
    validation_set = grouped_df[grouped_df['cell_line'].isin(validation_cell_lines)]
    training_set = grouped_df[~grouped_df['cell_line'].isin(validation_cell_lines)]
else:
    validation_set = grouped_df.sample(frac=0.2)
    training_set = grouped_df.drop(validation_set.index)

training_set = training_set[['chr', 'start', 'end', 'cell_line', 'label', 'count','region_type_enhancer', 'region_type_promoter']]
validation_set = validation_set[['chr', 'start', 'end', 'cell_line', 'label', 'count','region_type_enhancer', 'region_type_promoter']]

training_set_file = os.path.join(os.path.dirname(output_file), 'training_combined.csv')
validation_set_file = os.path.join(os.path.dirname(output_file), 'validation_combined.csv')
training_set.to_csv(training_set_file, sep='\t', index=False)
validation_set.to_csv(validation_set_file, sep='\t', index=False)

logging.info(f"Validation set generated with: {validation_cell_lines}")
logging.info(f"Training set generated with cell lines: {training_set['cell_line'].value_counts().index.tolist()}")

logging.info(f"Training set saved to: {training_set_file}")
logging.info(f"Validation set saved to: {validation_set_file}")