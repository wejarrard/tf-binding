import glob
import os

import pandas as pd


def calculate_accuracy(unbound_df, bound_df, tf, cell_line):
    total_predictions = len(unbound_df)
    true_negatives = unbound_df[10].value_counts().get(0, 0)
    false_positives = unbound_df[10].value_counts().get(1, 0)

    total_predictions += len(bound_df)
    true_positives = bound_df[10].value_counts().get(1, 0)
    false_negatives = bound_df[10].value_counts().get(0, 0)

    # Accuracy calculation
    accuracy = (true_negatives + true_positives) / total_predictions

    # Update the summary DataFrame
    summary_df = pd.DataFrame(
        {
            "tf": [tf],
            "cell_line": [cell_line],
            "accuracy": [accuracy],
            "total_predictions": [total_predictions],
            "false_positives": [false_positives],
            "true_negatives": [true_negatives],
            "false_negatives": [false_negatives],
            "true_positives": [true_positives],
        }
    )

    return summary_df


def get_tobias_results(directory):
    summary_dfs = []  # List to store summary DataFrames for each tf

    for tf in os.listdir(directory):
        tf_dir = f"{directory}/{tf}/tobias"
        if not os.path.exists(tf_dir):
            print(f"Directory '{tf_dir}' does not exist")
            continue
        for cell_line in os.listdir(tf_dir):
            cell_line_dir = f"{tf_dir}/{cell_line}"
            matching_dirs = [dir for dir in os.listdir(cell_line_dir) if tf in dir]
            if len(matching_dirs) == 0:
                print(
                    f"No matching directories found for {tf} in {cell_line_dir}, only options are {os.listdir(cell_line_dir)}"
                )
                continue
            if len(matching_dirs) > 1:
                # choose one that ends with A
                matching_dirs = [dir for dir in matching_dirs if dir.endswith("A")]
            if len(matching_dirs) != 1:
                print(
                    f"Expected exactly one matching directory for {tf} in {cell_line_dir}, found {matching_dirs}"
                )
                continue
            matching_dir = matching_dirs[0]
            # Construct the path pattern to match files ending with '_unbound.bed'
            pattern = f"{cell_line_dir}/{matching_dir}/beds/*_unbound.bed"

            # Use glob to find files matching the pattern
            unbound_files = glob.glob(pattern)

            # Ensure there is exactly one matching file
            if len(unbound_files) == 0:
                print(
                    f"No unbound files found for {tf} in {cell_line_dir}/{matching_dir}/beds"
                )
                continue

            assert (
                len(unbound_files) == 1
            ), f"Expected exactly one file ending with '_unbound.bed', found {unbound_files}"

            # Read the CSV file
            unbound_df = pd.DataFrame()
            if os.path.getsize(unbound_files[0]) > 0:
                unbound_df = pd.read_csv(unbound_files[0], sep="\t", header=None)
            else:
                print(f"Empty file found for {tf} in {unbound_files[0]}")
                continue

            # Construct the path pattern to match files ending with '_bound.bed'
            pattern = f"{cell_line_dir}/{matching_dir}/beds/*_bound.bed"

            # Use glob to find files matching the pattern
            bound_files = glob.glob(pattern)

            # Ensure there is exactly one matching file
            assert (
                len(bound_files) == 1
            ), "Expected exactly one file ending with '_bound.bed'"

            # Read the CSV file
            bound_df = pd.DataFrame()
            if os.path.getsize(bound_files[0]) > 0:
                bound_df = pd.read_csv(bound_files[0], sep="\t", header=None)
            else:
                print(f"Empty file found for {tf} in {bound_files[0]}")
                continue

            summary_df = calculate_accuracy(unbound_df, bound_df, tf, cell_line)
            summary_dfs.append(summary_df)  # Append the summary DataFrame to the list

    # Concatenate all the summary DataFrames into a single DataFrame
    summary_df = pd.concat(summary_dfs, ignore_index=True)

    return summary_df


def main():

    directory = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors"

    results = get_tobias_results(directory)

    results.to_csv("tobias_results.csv", index=False)


if __name__ == "__main__":
    main()

