### NEGATIVE TEST
import os

import polars as pl


def read_peak_file(filepath):
    return pl.read_csv(
        filepath,
        separator="\t",
        has_header=False,
        columns=[0, 1, 2],
        new_columns=["Chromosome", "Start", "End"],
    )


def has_no_overlap(df_row, peak_df, label):
    adjusted_start = max(df_row["End"] - 2000, 0)
    adjusted_end = df_row["Start"] + 2000

    filtered_df = peak_df.filter(
        (peak_df["Chromosome"] == df_row["Chromosome"])
        & (peak_df["Start"] < adjusted_end)
        & (peak_df["End"] > adjusted_start)
    )
    if filtered_df.height > 0:
        print("_____")
        print(label)
        print(df_row)
        print(filtered_df)

    return not (filtered_df.height > 0)


def check_no_overlap_for_sampled_rows(df, cell_lines_directory, sample_size):
    sampled_df = df.sample(n=sample_size)
    results = []

    for row in sampled_df.iter_rows(named=True):
        labels = row["labels"].split(",")
        for label in labels:
            folder_path = os.path.join(cell_lines_directory, label, "peaks")
            for filename in os.listdir(folder_path):
                if filename.endswith("filtered.broadPeak"):
                    file_path = os.path.join(folder_path, filename)
                    # Check if file is empty by attempting to read the first byte
                    if os.path.getsize(file_path) == 0:
                        # If the file is empty, skip this label
                        continue
                    peak_df = read_peak_file(file_path)
                    # Check if peak_df is empty
                    if peak_df.height == 0:
                        # If the DataFrame is empty, skip this label
                        continue
                    # Invert the result of has_overlap for the negative check
                    results.append(has_no_overlap(row, peak_df, label))

    return all(results)


def test_negative_data_validation():
    df = pl.read_csv(
        "data/negative.bed",
        separator="\t",
        has_header=False,
        new_columns=["Chromosome", "Start", "End", "source", "labels"],
    )
    cell_lines_directory = (
        "/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines"
    )
    sample_size = 100

    # The assertion will fail if there's an overlap for any sampled row
    assert check_no_overlap_for_sampled_rows(
        df, cell_lines_directory, sample_size
    ), "One or more cell lines have an overlap with our consolidated file for one or more sampled rows"

    # assert not check_overlap_for_sampled_rows(
    #     df, cell_lines_directory, sample_size
    # ), "One or more cell lines do not have an overlap with our consolidated file for one or more sampled rows"
