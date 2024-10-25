import pandas as pd
import pybedtools
import os
import glob
from sklearn.model_selection import train_test_split

tf1 = "FOXA1"
tf2 = "AR"

cell_line_mapping = {
    "293": "HEK_293",
    "A549": "A549",
    "ECC-1": "AN3_CA",
    "HUH-7": "HuH-7",
    "K-562": "K562",
    "MCF-7": "MCF7",
    "22RV1": "22Rv1",
    "A-375": "A-375",
    "C4-2": "C4-2",
    "LNCAP": "LNCAP",
    "PC-3": "PC-3",
    "THP-1": "THP-1",
    "UM-UC-3": "UM-UC-3",
    "VCAP": "VCAP",
    "CAMA-1": "CAMA-1",
    "HCC1954": "HCC1954",
    "HEC-1-A": "HEC-1-A",
    "ISHIKAWA": "Ishikawa",
    "MDA-MB-134-VI": "MDA-MB-134-VI",
    "MDA-MB-453": "MDA-MB-453",
    "NCI-H3122": "NCI-H3122",
    "NCI-H441": "NCI-H441",
    "RT4": "RT4",
    "SK-BR-3": "SK-BR-3",
    "GP5D": "GP5D",
    "LS-180": "LS-180",
    "CFPAC-1": "CFPAC-1",
    "GM12878": "GM12878",
    "MDA-MB-231": "MDA-MB-231",
    "HELA": "HELA",
    "SK-N-SH": "SK-N-SH",
    "HMEC": "HMEC",
    "U2OS": "U2OS"
}


# Function to get the folder name from the TF1 file name
def get_folder_name(tf1_file):
    cell_line = os.path.basename(tf1_file).replace(f"_{tf1}.bed", "")
    return cell_line_mapping.get(cell_line, None)

# Function to filter DataFrame based on given criteria
def filter_df(df, threshold=4000):
    df = df[df['end'] - df['start'] <= threshold]
    if df['count'].max() > 50:
        return df.query('count >= @df["count"].quantile(.75)')
    elif df['count'].max() <= 2:
        return df
    else:
        return df.query('count >= @df["count"].median()')

# Directory containing the BED files
directory = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/{tf1}_{tf2}"
input_directory = f"{directory}/output"

# List all TF1 and TF2 files
tf1_files = glob.glob(os.path.join(input_directory, f"*_{tf1}.bed"))
tf2_files = glob.glob(os.path.join(input_directory, f"*_{tf2}.bed"))

# List to store all DataFrames
combined_data = []

# Process each pair of TF1 and TF2 files
for tf1_file in tf1_files:
    # Extract the cell line name from the file name
    cell_line = os.path.basename(tf1_file).replace(f"_{tf1}.bed", "")
    tf2_file = os.path.join(input_directory, f"{cell_line}_{tf2}.bed")

    cell_line = get_folder_name(tf1_file)

    if cell_line is None:
        continue
    
    # Check if the corresponding TF2 file exists
    if os.path.exists(tf2_file):
        # Read the BED files into DataFrames
        tf1_df = pd.read_csv(tf1_file, sep="\t", header=None, names=['chr', 'start', 'end', 'count'])
        tf2_df = pd.read_csv(tf2_file, sep="\t", header=None, names=['chr', 'start', 'end', 'count'])

        # Filter the DataFrames
        tf1_filtered = filter_df(tf1_df)
        tf2_filtered = filter_df(tf2_df)

        # Convert the filtered DataFrames to BedTool objects
        tf1_bedtool = pybedtools.BedTool.from_dataframe(tf1_filtered)
        tf2_bedtool = pybedtools.BedTool.from_dataframe(tf2_filtered)

        # Perform the subtraction
        tf1_only = tf1_bedtool.subtract(tf2_bedtool, A=True)
        tf2_only = tf2_bedtool.subtract(tf1_bedtool, A=True)

        # Perform the intersection
        intersection = tf1_bedtool.intersect(tf2_bedtool, u=True)

        # Convert the results back to DataFrames
        tf1_only_df = tf1_only.to_dataframe()
        tf2_only_df = tf2_only.to_dataframe()

        # Add cell line column to each DataFrame
        tf1_only_df['cell_line'] = cell_line
        tf2_only_df['cell_line'] = cell_line

        # Add label column to each DataFrame
        tf1_only_df['label'] = 1
        tf2_only_df['label'] = 0

        # Combine all DataFrames into one
        combined_data.append(tf1_only_df)
        combined_data.append(tf2_only_df)

# Concatenate all data
combined_df = pd.concat(combined_data)

# Calculate the total number of samples in each class
label_counts = combined_df['label'].value_counts()
majority_class_count = label_counts.max()
minority_class_count = label_counts.min()

# Calculate the weight for the minority class
minority_class_weight = majority_class_count / minority_class_count

# Add weight column
combined_df['weight'] = combined_df['label'].apply(lambda x: 1 if x == 1 else minority_class_weight)

combined_df = combined_df.rename(columns={'chrom': 'chr', "weight": "count"})[['chr', 'start', 'end', 'cell_line', 'count', 'label']]

# Separate the data into two subsets based on the class label
tf1_df = combined_df[combined_df['label'] == 1]
tf2_df = combined_df[combined_df['label'] == 0]

# Determine the number of samples for the balanced validation set
val_size_per_class = min(len(tf1_df), len(tf2_df)) // 10

# Split each subset into training and validation sets
tf1_train, tf1_val = train_test_split(tf1_df, test_size=val_size_per_class, random_state=42)
tf2_train, tf2_val = train_test_split(tf2_df, test_size=val_size_per_class, random_state=42)

# Concatenate the validation sets to form a balanced validation set
val_df = pd.concat([tf1_val, tf2_val])

# Concatenate the remaining data to form the training set
train_df = pd.concat([tf1_train, tf2_train])

# Save training and validation sets to files
os.makedirs(os.path.join(directory, f"data_splits"))
train_df.to_csv(os.path.join(directory, f"data_splits/train_data.bed"), sep="\t", header=True, index=False)
val_df.to_csv(os.path.join(directory, "data_splits/val_data.bed"), sep="\t", header=True, index=False)

# Print the number of regions in training and validation sets
print(f"Training set regions: {len(train_df)}")
print(f"Validation set regions: {len(val_df)}")
