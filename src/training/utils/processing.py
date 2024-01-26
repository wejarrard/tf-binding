import polars as pl
import torch

def get_weights(bed_path):
    # Read the CSV file
    df = pl.read_csv(bed_path, separator="\t", has_header=False, 
                     new_columns=["Chromosome", "Start", "End", "source", "labels"])

    # Count the occurrences of each label
    value_counts = df.groupby('labels').count()

    # Create a dictionary from the value_counts DataFrame
    counts_dict = value_counts.to_dict(as_series=False)

    # Extract counts for Negative and Positive
    negative_count = counts_dict['count'][counts_dict['labels'].index('Negative')]
    positive_count = counts_dict['count'][counts_dict['labels'].index('Positive')]

    # Total number of samples
    total_samples = len(df)

    # Calculate weights inversely proportional to class frequencies
    negative_weight = total_samples / (2 * negative_count)
    positive_weight = total_samples / (2 * positive_count)

    # Create a tensor for weights
    weights = torch.tensor([negative_weight, positive_weight], dtype=torch.float)

    # Print the weights
    print(f"Negative weight: {weights[0]}")
    print(f"Positive weight: {weights[1]}")

    return weights




if __name__ == "__main__":
    weights = get_weights("data/AR_ATAC_broadPeak")
    print(weights)
