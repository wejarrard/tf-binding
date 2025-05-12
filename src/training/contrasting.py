import pandas as pd
import numpy as np
import os
import tempfile
import subprocess
import json

from sagemaker import Session
from sagemaker.pytorch import PyTorch


# ============================================================
TF1 = "FOXA1"
TF2 = "FOXA2"
OUTPUT_DIR = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/data_splits"

# Validation set configuration
VALIDATION_TYPE = "cell_lines"  # Options: "chromosomes" or "cell_lines"
VALIDATION_CHROMOSOMES = ["chr7", "chr11"]  # Chromosomes to use for validation
VALIDATION_CELL_LINES = ["Hep_G2"]  # Cell lines to use for validation

# ============================================================
# Constants
# ============================================================
VALID_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
LENGTH_THRESHOLD = 4000
HIGH_COUNT_QUANTILE = 0.5
MAX_COUNT_THRESHOLD = 30
MID_COUNT_THRESHOLD = 10

def load_cell_line_mapping(file_path: str) -> dict:
    with open(file_path, "r") as file:
        return json.load(file)

cell_line_mapping = load_cell_line_mapping("/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/utils/cell_line_mapping.json")

# ============================================================
# Utility Functions
# ============================================================

def run_bedtools_command(command: str) -> None:
    subprocess.run(command, shell=True, check=True) 

def drop_duplicates_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().sort_values(by=["chr", "start", "end"]).reset_index(drop=True)

def intersect_bed_files(main_df: pd.DataFrame, intersect_df: pd.DataFrame, region_type: str = None) -> pd.DataFrame:
    """
    Intersect two BED files using bedtools and return the original DataFrame with overlap flags.
    
    Args:
        main_df: Primary pandas DataFrame with BED data
        intersect_df: Secondary pandas DataFrame to intersect with
        region_type: Optional region type label to add to results
        
    Returns:
        Original DataFrame with additional column indicating overlaps
    """
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as main_file, \
         tempfile.NamedTemporaryFile(delete=False, mode='w') as intersect_file, \
         tempfile.NamedTemporaryFile(delete=False, mode='w') as result_file:
        
        main_path = main_file.name
        intersect_path = intersect_file.name
        result_path = result_file.name

        # Write DataFrames to temporary files
        main_df.to_csv(main_path, sep="\t", header=False, index=False)
        intersect_df.to_csv(intersect_path, sep="\t", header=False, index=False)

        # Run bedtools intersect with -c flag to count overlaps
        command = f"bedtools intersect -a {main_path} -b {intersect_path} -c > {result_path}"
        run_bedtools_command(command)

        # Read results back into pandas DataFrame
        result_df = pd.read_csv(
            result_path,
            sep="\t",
            header=None,
            names=[*main_df.columns, "overlap_count"]
        )

    # Clean up temporary files
    os.remove(main_path)
    os.remove(intersect_path) 
    os.remove(result_path)

    # Add boolean overlap column
    result_df["overlaps_ground_truth"] = result_df["overlap_count"] > 0
    result_df = result_df.drop("overlap_count", axis=1)

    return result_df


def filter_by_chromosomes(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["chr"].isin(VALID_CHROMOSOMES)]

def filter_by_peak_length(df: pd.DataFrame, threshold: int = LENGTH_THRESHOLD) -> pd.DataFrame:
    return df[(df["end"] - df["start"]) <= threshold]



def threshold_peaks(df):
    max_count = df["count"].max()
    
    if max_count <= 2:
        return df
    elif max_count > MAX_COUNT_THRESHOLD:
        threshold = df["count"].quantile(HIGH_COUNT_QUANTILE)
        df = df[df["count"] > threshold]
    elif max_count > MID_COUNT_THRESHOLD:
        threshold = df["count"].median()
        df = df[df["count"] > threshold]
    return df

# ============================================================
# Main
# ============================================================


# cell lines for TF1
TF1_DIR = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/transcription_factors/{TF1}/merged"
TF1_CELL_LINES = [cell_line.split("_")[0] for cell_line in os.listdir(TF1_DIR)]


# cell lines for TF2
TF2_DIR = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/transcription_factors/{TF2}/merged"
TF2_CELL_LINES = [cell_line.split("_")[0] for cell_line in os.listdir(TF2_DIR)]

# get list of cell lines with both TF1 and TF2 peaks
CELL_LINES = list(set(TF1_CELL_LINES) & set(TF2_CELL_LINES))




contrasting_dfs = []

for cell_line in CELL_LINES:

    atac_path = f"/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/{cell_line_mapping[cell_line]}/peaks/{cell_line_mapping[cell_line]}.filtered.broadPeak"
    atac_df = pd.read_csv(atac_path, sep="\t", header=None, names=["chr", "start", "end"], usecols=[0, 1, 2])
    atac_df = filter_by_peak_length(atac_df)
    atac_df = filter_by_chromosomes(atac_df)
        # get peaks for TF1
    TF1_PEAKS = pd.read_csv(f"{TF1_DIR}/{cell_line}_{TF1}_merged.bed", sep="\t", header=None, names=["chr", "start", "end", "count"])
    TF1_PEAKS = threshold_peaks(TF1_PEAKS)
    TF1_atac = intersect_bed_files(atac_df, TF1_PEAKS)


    # get peaks for TF2
    TF2_PEAKS = pd.read_csv(f"{TF2_DIR}/{cell_line}_{TF2}_merged.bed", sep="\t", header=None, names=["chr", "start", "end", "count"])
    TF2_PEAKS = threshold_peaks(TF2_PEAKS)
    TF2_atac = intersect_bed_files(atac_df, TF2_PEAKS)
        
    contrasting_df = pd.merge(
        TF1_atac.rename(columns={'overlaps_ground_truth': f'{TF1}'}),
        TF2_atac.rename(columns={'overlaps_ground_truth': f'{TF2}'}),
        on=["chr", "start", "end"],
        how="inner"
    )

    # filter for peaks that are only in one of the two TFs
    contrasting_df = contrasting_df[
        (contrasting_df[TF1] == False) & (contrasting_df[TF2] == True) | 
        (contrasting_df[TF1] == True) & (contrasting_df[TF2] == False)
    ]

    # add binding pattern column
    contrasting_df['label'] = contrasting_df.apply(
        lambda row: 1 if row[TF1] else 0,
        axis=1
    )

    # drop the TF columns
    contrasting_df = contrasting_df.drop([TF1, TF2], axis=1)
    contrasting_df['cell_line'] = cell_line_mapping[cell_line]
    contrasting_df['count'] = contrasting_df['label']

    contrasting_df = contrasting_df[['chr', 'start', 'end', 'count', 'label', 'cell_line']]

    contrasting_dfs.append(contrasting_df)

contrasting_df = pd.concat(contrasting_dfs)

contrasting_df['cell_line'].value_counts()

# get len of 0 and 1 in contrasting_df
len_0 = len(contrasting_df[contrasting_df['label'] == 0])
len_1 = len(contrasting_df[contrasting_df['label'] == 1])

print(f"Length of 0: {len_0}")
print(f"Length of 1: {len_1}")

# create validation set based on configuration
if VALIDATION_TYPE == "chromosomes":
    validation_df = contrasting_df[contrasting_df['chr'].isin(VALIDATION_CHROMOSOMES)]
    training_df = contrasting_df[~contrasting_df['chr'].isin(VALIDATION_CHROMOSOMES)]
else:  # cell_lines
    validation_df = contrasting_df[contrasting_df['cell_line'].isin(VALIDATION_CELL_LINES)]
    training_df = contrasting_df[~contrasting_df['cell_line'].isin(VALIDATION_CELL_LINES)]

# double check lengths of binding patterns
len_0_validation = len(validation_df[validation_df['label'] == 0])
len_1_validation = len(validation_df[validation_df['label'] == 1])

print(f"Length of 0 in validation: {len_0_validation}")
print(f"Length of 1 in validation: {len_1_validation}")

# save the contrasting df
validation_df.to_csv(f"{OUTPUT_DIR}/validation_{TF1}_{TF2}.csv", sep="\t", header=False, index=False)

# double check lengths of binding patterns
len_0_training = len(training_df[training_df['label'] == int(0)])
len_1_training = len(training_df[training_df['label'] == int(1)])

print(f"Length of 0 in training: {len_0_training}")
print(f"Length of 1 in training: {len_1_training}")

# save the training set
training_df.to_csv(f"{OUTPUT_DIR}/training_{TF1}_{TF2}.csv", sep="\t", index=False)




# ============================================================
# Upload data to S3
# ============================================================

print("Uploading data to S3...")
sagemaker_session = Session()
local_dir = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/data_splits"
inputs = sagemaker_session.upload_data(path=local_dir, bucket="tf-binding-sites", key_prefix="pretraining/data")

estimator = PyTorch(
    base_job_name=f"{TF1}-{TF2}-Contrasting",
    entry_point='tf_prediction.py',
    source_dir="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/training/tf_finetuning",
    output_path=f"s3://tf-binding-sites/finetuning/results/output",
    code_location=f"s3://tf-binding-sites/finetuning/results/code",
    role='arn:aws:iam::016114370410:role/tf-binding-sites',
    py_version="py310",
    framework_version='2.0.0',
    volume_size=900,
    instance_count=1,
    max_run=1209600,
    instance_type='ml.g5.16xlarge',
    hyperparameters={
        'learning-rate': 1e-5,
        'train-file': f'training_{TF1}_{TF2}.csv',
        'valid-file': f'validation_{TF1}_{TF2}.csv',
    }
)
        
estimator.fit(inputs, wait=False)