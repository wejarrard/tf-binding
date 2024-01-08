import os

import polars as pl
from polars import col
from tqdm import tqdm


def count_peaks_over_window_size(path, window_size=16384):
    df = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=["Chromosome", "Start", "End", "source", "labels"],
    )
    print(df.height)

    # Filter the DataFrame based on the condition and count the resulting rows
    df_filtered = df.filter(col("End") - col("Start") <= window_size)
    print(df_filtered.height)

    return df_filtered


# ################# Filter non compliant rows #####################

# def has_no_overlap(df_row, peak_df):
#     adjusted_start = max(df_row["End"] - 8_192, 0)
#     adjusted_end = df_row["Start"] + 8_192

#     filtered_df = peak_df.filter(
#         (peak_df["Chromosome"] == df_row["Chromosome"])
#         & (peak_df["Start"] < adjusted_end)
#         & (peak_df["End"] > adjusted_start)
#     )

#     return not filtered_df.height > 0

# def read_peak_file(file_path):
#     # Assuming the peak file has a specific format, adjust as needed
#     return pl.read_csv(
#         file_path,
#         separator="\t",
#         has_header=False,
#         new_columns=["Chromosome", "Start", "End", "source", "labels"]
#     )

# def check_no_overlap_for_all_rows(df, cell_lines_directory):
#     no_overlap_rows = []

#     for row in tqdm(df.iter_rows(named=True), total=df.height, desc="Processing rows"):
#         labels = row["labels"].split(",")
#         no_overlap = True
#         for label in labels:
#             folder_path = os.path.join(cell_lines_directory, label, "peaks")
#             for filename in os.listdir(folder_path):
#                 if filename.endswith("filtered.broadPeak"):
#                     file_path = os.path.join(folder_path, filename)
#                     if os.path.getsize(file_path) == 0:
#                         continue
#                     peak_df = read_peak_file(file_path)
#                     if peak_df.height == 0:
#                         continue
#                     if not has_no_overlap(row, peak_df):
#                         no_overlap = False
#                         break
#             if not no_overlap:
#                 break
#         if no_overlap:
#             no_overlap_rows.append(row)

#     df_filtered = pl.DataFrame(no_overlap_rows)
#     return df_filtered


if __name__ == "__main__":
    df_filtered = count_peaks_over_window_size("../../AR_ATAC_broadPeak")
    df_filtered.write_csv("./AR_ATAC_broadPeak", separator="\t", has_header=False)

    # negative = read_peak_file("data/negative.bed")

    # negative_filtered = check_no_overlap_for_all_rows(negative, "/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines")

    # negative_filtered.write_csv("data/negative_filtered.bed", separator="\t", has_header=False)
