import pandas as pd
import os
import numpy as np
import argparse
import json
import logging
from utils.bedtools import subtract_bed_files, intersect_bed_files, intersect_colocalization_bed_files

# Set up argument parser
parser = argparse.ArgumentParser(description="Process TF binding data.")

parser.add_argument("tf", type=str, help="Transcription Factor")
parser.add_argument('--balance', action='store_true', help='Balance the labels in the datasets')
parser.add_argument('--enhancer_promotor_only', action='store_true', help='Only consider enhancer and promotor regions')
parser.add_argument('--dont_filter', action='store_true', help='Dont filter the data')

# Add the two mutually exclusive arguments
validation_group = parser.add_mutually_exclusive_group(required=False)
validation_group.add_argument("--validation_cell_lines", type=str, nargs='*', help="Cell lines for validation set")
validation_group.add_argument("--validation_chromosomes", type=str, nargs='*', help="Chromosomes for validation set")

group = parser.add_mutually_exclusive_group(required=False)
group.add_argument('--negative_tf', type=str, help='Transcription factor for negative dataset')
group.add_argument('--colocalization_tf', type=str, help='Transcription factor for colocalization dataset')

parser.add_argument('--chip_provided', action='store_true', help='Use the provided chip data for the negative set')

args = parser.parse_args()

# Validate that --chip_provided is only used if --colocalization_tf is also provided
if args.chip_provided and not args.colocalization_tf:
    parser.error("--chip_provided requires --colocalization_tf to be provided")

if args.negative_tf:
    NotImplementedError("The feature 'negative_tf' is not implemented yet")

tf = args.tf
balance = args.balance
enhancer_promotor_only = args.enhancer_promotor_only
negative_tf = args.negative_tf if args.negative_tf else None
colocalization_tf = args.colocalization_tf if args.colocalization_tf else None
dont_filter = args.dont_filter
chip_provided = args.chip_provided

if args.validation_cell_lines:
    validation_cell_lines = args.validation_cell_lines
    validation_chromosomes = None
    logging.info(f"Validation cell lines: {validation_cell_lines}")
elif args.validation_chromosomes:
    validation_chromosomes = args.validation_chromosomes
    validation_cell_lines = None
    logging.info(f"Validation chromosomes: {validation_chromosomes}")
else:
    validation_cell_lines = None
    validation_chromosomes = None
    logging.info("No validation set specified, using default 20% \split")


######################################################
input_dir = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{tf}/output"
output_file = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{tf}/entire_set/output_{tf}.csv"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_df(df, threshold=4000, drop_rows=True):
    try:
        # Applying the threshold filter by dropping rows
        df = df[df['end'] - df['start'] <= threshold]
        df['label'] = 1
        if drop_rows:
            if df['count'].max() <= 2:
                return df
            # elif df['count'].max() > 5:
            #     return df.query('count > @df["count"].median()')
            else:
                return df.query('count >= @df["count"].median()')
        else:
            median_count = df['count'].median()
            if df['count'].max() <= 2:
                pass
            # elif df['count'].max() > 5:
            #     df.loc[df['count'] <= median_count, 'label'] = 0
            else:
                df.loc[df['count'] < median_count, 'label'] = 0
        return df
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
                if balance:
                    filtered_df = filter_df(df, drop_rows=True)
                else:
                    filtered_df = filter_df(df, drop_rows=False)
                
            filtered_df = filter_chromosomes(filtered_df).copy()
            cell_line_key = filename.split('_')[0]
            cell_line = cell_lines.get(cell_line_key)
            neg_df = pd.DataFrame()

            if cell_line:
                filtered_df['cell_line'] = cell_line
                if colocalization_tf:
                    colocalization_filepath = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{colocalization_tf}/output/{cell_line_key}_{colocalization_tf}.bed"
                    atac_filepath = f"/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/{cell_line}/peaks/{cell_line}.filtered.broadPeak"
                    if os.path.exists(colocalization_filepath):
                        # second tf that we want colocalization with
                        colocalization_df = pd.read_csv(colocalization_filepath, sep="\t", usecols=[0, 1, 2, 3], header=None, names=['chr', 'start', 'end', 'count'])

                        if dont_filter:
                            colocalization_df = colocalization_df
                        else:
                            if balance:
                                colocalization_df = filter_df(colocalization_df, drop_rows=True)
                            else:
                                # TODO
                                colocalization_df = filter_df(colocalization_df, drop_rows=True)
                        # Entire atacseq dataset
                        atac_df = pd.read_csv(atac_filepath, sep="\t", usecols=[0, 1, 2], header=None, names=['chr', 'start', 'end'])
                        atac_df = filter_chromosomes(atac_df).copy()

                        # these make sure we keep atac seq chr start end values for consistency
                        tf1_df = intersect_colocalization_bed_files(filtered_df, atac_df)
                        tf2_df = intersect_colocalization_bed_files(colocalization_df, atac_df)

                        # Now we want a df with no tfs in it

                        # this is so we can see which samples are only in one of the tfs, purely for double checking unless we are using model with chip input
                        tf1_only_df = subtract_bed_files(tf1_df, tf2_df[['chr', 'start', 'end']])
                        tf2_only_df = subtract_bed_files(tf2_df, tf1_df[['chr', 'start', 'end']])

                        if chip_provided:
                            neg_df = tf1_only_df
                        else:
                            neg_df = subtract_bed_files(atac_df, tf1_df[['chr', 'start', 'end']])
                            neg_df = subtract_bed_files(neg_df, tf2_df[['chr', 'start', 'end']])
                        
                        # print(tf1_only_df)
                        # print(tf2_only_df)

                        # We want to get the overlapping samples between the tfs, and this should now have the atac seq values for chr start end
                        overlapping_df = intersect_colocalization_bed_files(tf1_df, tf2_df[['chr', 'start', 'end', 'count']], count_included=True)
                        # TODO Figure out why reverse is very slightyl different only for overlapping df
                        # we are also currently losing count for tf2, which could be important
                        # print(f"overlapping_df: {len(overlapping_df)}")

                        # TODO Keep the count for the second tf, add in storing if only one tf is present

                        
                        overlapping_df['cell_line'] = cell_line
                        overlapping_df['label'] = 1

                        tf1_only_df['label'] = 0
                        tf2_only_df['label'] = 0


                        neg_df['cell_line'] = cell_line
                        neg_df['label'] = 0
                        neg_df['count'] = 0
                        neg_df['count_2'] = 0

                        filtered_dfs.append(overlapping_df)
                        if balance:
                            filtered_dfs.append(tf1_only_df)
                            filtered_dfs.append(tf2_only_df)
                            # append neg samples of length equal to min of tf1_only_df + tf2_only_df if tf1_only_df + tf2_only_df > overlapping_df else length of overlapping_df
                            filtered_dfs.append(neg_df.sample(min(len(tf1_only_df) + len(tf2_only_df), len(overlapping_df))))
                        else:
                            filtered_dfs.append(tf1_only_df)
                            filtered_dfs.append(tf2_only_df)
                            filtered_dfs.append(neg_df)

                        logging.info(f"Cell line: {cell_line}; Colocalization samples: {len(overlapping_df)} overlapping samples, {len(neg_df)} negative samples, this includes {len(tf1_only_df)} {tf} only samples and {len(tf2_only_df)} {colocalization_tf} only samples")
                    else:
                        logging.warning(f"Colocalization samples file not found for cell line {cell_line} at {colocalization_filepath}")
                else:
                    negative_filepath = f"/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/{cell_line}/peaks/{cell_line}.filtered.broadPeak"
                    if os.path.exists(negative_filepath):
                        neg_df = pd.read_csv(negative_filepath, sep="\t", usecols=[0, 1, 2], header=None, names=['chr', 'start', 'end'])
                        neg_df = filter_chromosomes(neg_df).copy()
                        # print(f"filtered negative samples: {len(neg_df)}")
                        neg_df = subtract_bed_files(neg_df, filtered_df[['chr', 'start', 'end']])
                        # print(f"subtracted negative samples: {len(neg_df)}")

                        neg_df['count'] = 0
                        neg_df['cell_line'] = cell_line
                        neg_df['label'] = 0

                        #TODO if sum(filtered_df['label'] == 1) == 0 then dont add to filtered_dfs

                        filtered_dfs.append(filtered_df)
                        filtered_dfs.append(neg_df)
                        logging.info(f"{filename}: {sum(filtered_df['label'] == 1)} positive samples, {len(neg_df) + sum(filtered_df['label'] == 0)} negative samples")
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

# TODO: Could add in test set option, as well as the option to just sample for the validation set
# TODO: Could oversample minority class instead of undersampling majority class to keep data
if balance:
    if colocalization_tf:
        grouped_df = grouped_df.groupby('cell_line').apply(balance_labels, include_groups=False).reset_index(drop=False).drop(columns="level_1")
    if negative_tf:
        grouped_df = grouped_df.groupby('cell_line').apply(balance_labels, include_groups=False).reset_index(drop=False).drop(columns="level_1")
    else:
        grouped_df = grouped_df.groupby('cell_line').apply(balance_labels, include_groups=False).reset_index(drop=False).drop(columns="level_1")


# Split the dataset into training and validation sets
if validation_cell_lines:
    validation_set = grouped_df[grouped_df['cell_line'].isin(validation_cell_lines)]
    training_set = grouped_df[~grouped_df['cell_line'].isin(validation_cell_lines)]
elif validation_chromosomes:
    validation_set = grouped_df[grouped_df['chr'].isin(validation_chromosomes)]
    training_set = grouped_df[~grouped_df['chr'].isin(validation_chromosomes)]
else:
    validation_set = grouped_df.sample(frac=0.2)
    training_set = grouped_df.drop(validation_set.index)

if colocalization_tf:
    validation_set = validation_set[['chr', 'start', 'end', 'cell_line', 'label', 'count', 'count_2','region_type_enhancer', 'region_type_promoter']]
    training_set = training_set[['chr', 'start', 'end', 'cell_line', 'label', 'count', 'count_2', 'region_type_enhancer', 'region_type_promoter']]
else:
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