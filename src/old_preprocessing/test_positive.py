import os

import polars as pl

### POSITIVE TEST


def read_peak_file(filepath):
    return pl.read_csv(
        filepath,
        separator="\t",
        has_header=False,
        columns=[0, 1, 2],
        new_columns=["Chromosome", "Start", "End"],
    )


def has_overlap(df_row, peak_df, label, index, original_df):
    adjusted_start = df_row["Start"]
    adjusted_end = df_row["End"]

    filtered_df = peak_df.filter(
        (peak_df["Chromosome"] == df_row["Chromosome"])
        & (peak_df["Start"] <= adjusted_end)
        & (peak_df["End"] >= adjusted_start)
    )

    if filtered_df.height == 0:
        print("_____")
        print("Label:", label)
        print("Current Row:", df_row)

        # Define the range of rows to display around the current index
        row_range = 4  # for example, 2 rows before and after
        start_idx = max(0, index - row_range)
        end_idx = min(len(original_df) - 1, index + row_range)

        surrounding_rows = original_df[start_idx : end_idx + 1]
        print("Surrounding Rows in Original DataFrame:")
        print(surrounding_rows)

    return filtered_df.height > 0


def check_overlap_for_sampled_rows(df, cell_lines_directory, sample_size):
    df = df.with_row_count("index")
    sampled_df = df.sample(n=sample_size)
    results = []

    for row in sampled_df.iter_rows(named=True):
        index = row["index"]
        labels = row["labels"].split(",")
        for label in labels:
            folder_path = os.path.join(cell_lines_directory, label, "peaks")
            for filename in os.listdir(folder_path):
                if filename.endswith("filtered.broadPeak"):
                    peak_df = read_peak_file(os.path.join(folder_path, filename))
                    results.append(has_overlap(row, peak_df, label, index, df))

    return all(results)


def test_positive_data_validation():
    df = pl.read_csv(
        "data/positive.bed",
        separator="\t",
        has_header=False,
        new_columns=["Chromosome", "Start", "End", "source", "labels"],
    )
    print(df)
    cell_lines_directory = (
        "/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines"
    )
    sample_size = 100

    # The assertion will fail if there's no overlap for any sampled row
    assert check_overlap_for_sampled_rows(
        df, cell_lines_directory, sample_size
    ), "One or more cell lines do not have an overlap with our consolidated file for one or more sampled rows"
