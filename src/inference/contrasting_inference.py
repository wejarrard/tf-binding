# create the bed file you want to predict on
import pandas as pd
import os
import tempfile
import subprocess
import json
import time


# ============================================================
TF1 = "FOXA2"
TF2 = "FOXM1"
model_name = "FOXA2-FOXM1-Contrasting"


# Validation set configuration
VALIDATION_TYPE = "chromosomes"  # Options: "chromosomes", "cell_lines"
VALIDATION_CELL_LINES = ["A549"]  # Cell lines to use for validation if VALIDATION_TYPE is "cell_lines"
VALIDATION_CHROMOSOMES = ["chr2"] # Chromosomes to use for validation if VALIDATION_TYPE is "chromosomes"
OUTPUT_DIR = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/data_splits"

if VALIDATION_TYPE == "cell_lines":
    assert len(VALIDATION_CELL_LINES) == 1, "VALIDATION_CELL_LINES must contain only one cell line if VALIDATION_TYPE is 'cell_lines'"
elif VALIDATION_TYPE == "chromosomes":
    assert len(VALIDATION_CHROMOSOMES) >= 1, "VALIDATION_CHROMOSOMES must contain at least one chromosome if VALIDATION_TYPE is 'chromosomes'"
else:
    raise ValueError("VALIDATION_TYPE must be either 'cell_lines' or 'chromosomes'")

# This variable is now determined dynamically later based on VALIDATION_TYPE
# CELL_LINE_DIR = f"/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/{VALIDATION_CELL_LINES[0]}"
BASE_CELL_LINE_DATA_DIR = "/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines"


# other 
PROJECT_PATH = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"
LOG_DIR = f"{PROJECT_PATH}/logs"


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

    # Print TF1 vs TF2 site counts for this cell line
    tf1_sites = len(contrasting_df[contrasting_df['label'] == 1])
    tf2_sites = len(contrasting_df[contrasting_df['label'] == 0])
    print(f"\nCell line {cell_line_mapping[cell_line]}:")
    print(f"Number of {TF1} sites: {tf1_sites}")
    print(f"Number of {TF2} sites: {tf2_sites}")

    contrasting_dfs.append(contrasting_df)

contrasting_df = pd.concat(contrasting_dfs)

contrasting_df['cell_line'].value_counts()



# create validation set based on configuration
if VALIDATION_TYPE == "cell_lines":
    test_df = contrasting_df[contrasting_df['cell_line'].isin(VALIDATION_CELL_LINES)]
elif VALIDATION_TYPE == "chromosomes":
    test_df = contrasting_df[contrasting_df['chr'].isin(VALIDATION_CHROMOSOMES)]
else:
    # This case should have been caught by the assertion earlier, but as a fallback:
    raise ValueError("Invalid VALIDATION_TYPE specified.")

# save the test set
test_df.to_csv(f"{OUTPUT_DIR}/test_{TF1}_{TF2}.csv", sep="\t", index=False)




import json

# load in models.json as a dictionary
with open("/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/inference/models.json", "r") as f:
    models = json.load(f)

# load in the model
model = models[model_name]

print(model)



# Determine validation target identifier and specific cell line directory for prepare_data.py
if VALIDATION_TYPE == "cell_lines":
    validation_target_identifier = VALIDATION_CELL_LINES[0]
    cell_line_dir_for_prepare_data = f"{BASE_CELL_LINE_DATA_DIR}/{VALIDATION_CELL_LINES[0]}"
elif VALIDATION_TYPE == "chromosomes":
    validation_target_identifier = "_".join(sorted(VALIDATION_CHROMOSOMES))
    cell_line_dir_for_prepare_data = BASE_CELL_LINE_DATA_DIR


#########################
# Prepare data using qsub
print(f"Preparing data for {validation_target_identifier} - {model_name} using qsub...")
    
    # Create qsub script content
qsub_script_content = f"""#!/bin/bash
#$ -N prep_{validation_target_identifier}_{model_name}
#$ -o {LOG_DIR}/contrasting_inference/{validation_target_identifier}_{model_name}.log
#$ -j y
#$ -l h_vmem=32G

source ~/.bashrc

conda activate pterodactyl

python "{PROJECT_PATH}/src/inference/prepare_data.py" \
    --input_file "{OUTPUT_DIR}/test_{TF1}_{TF2}.csv" \
    --output_path "{PROJECT_PATH}/data/jsonl/contrasting_{TF1}_{TF2}" \
    --cell_line_dir "{cell_line_dir_for_prepare_data}"
"""

# Write qsub script to file
qsub_script_path = f"{LOG_DIR}/contrasting_inference/{validation_target_identifier}_{model_name}.sh"
with open(qsub_script_path, 'w') as f:
    f.write(qsub_script_content)

# Make the script executable
os.chmod(qsub_script_path, 0o755)

# Submit the job and get job ID
qsub_output = subprocess.run(['qsub', qsub_script_path], capture_output=True, text=True)
job_id = qsub_output.stdout.strip().split()[2]

# Wait for the job to complete
print(f"Waiting for prepare_data.py job ({job_id}) to complete...")
while True:
    try:
        subprocess.run(['qstat', '-j', job_id], capture_output=True, check=True)
        time.sleep(30)
    except subprocess.CalledProcessError:
        break







# Run inference
print(f"Running inference for {validation_target_identifier} using {model_name}...")

# Create log file path
log_file = f"{LOG_DIR}/contrasting_inference/{validation_target_identifier}_{model_name}_inference.log"

# Run the inference script
inference_cmd = [
    "python",
    f"{PROJECT_PATH}/src/inference/aws_inference.py",
    "--model", model_name,
    "--sample", validation_target_identifier,
    "--model_paths_file", f"{PROJECT_PATH}/src/inference/models.json",
    "--project_path", PROJECT_PATH,
    "--local_dir", f"{PROJECT_PATH}/data/jsonl/contrasting_{TF1}_{TF2}"
]

# Run the command and capture output
with open(log_file, 'a') as f:
    process = subprocess.Popen(
        inference_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Stream output to both console and log file
    for line in process.stdout:
        print(line, end='')
        f.write(line)
        f.flush()
    
    process.wait()
