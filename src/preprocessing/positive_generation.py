import os
import warnings
from typing import Union

import polars as pl
from tqdm import tqdm
from utils import get_cell_line_labels

warnings.filterwarnings("ignore", category=FutureWarning)


def process_chromosome(chr: str, df: pl.DataFrame) -> pl.DataFrame:
    new_df_rows = []
    n = len(df)

    active_sources = {}

    last_source_index = 0

    for i in tqdm(range(n), desc=f"Processing {chr}"):
        row_i = df.row(i, named=True)
        start = row_i["Start"]
        end = row_i["End"]
        current_source = row_i["source"]

        # Update active sources: Remove sources that have ended before the current start
        active_sources = {
            source: end_val
            for source, end_val in active_sources.items()
            if end_val >= start
        }

        # Add the current source to active sources
        active_sources[current_source] = end

        # Determine overlapping sources
        overlapping_sources = [
            source for source, end_val in active_sources.items() if end_val >= start
        ]

        while last_source_index < n:
            temp_row = df.row(last_source_index, named=True)

            if temp_row["Start"] > end:
                break

            last_source_index += 1

        for j in range(i, last_source_index):
            row_j = df.row(j, named=True)
            if row_j["Start"] >= end:
                break
            if row_j["Chromosome"] == row_i["Chromosome"]:
                overlapping_sources.append(row_j["source"])

        # Create a new row if there are any overlapping sources
        overlapping_sources = list(set(overlapping_sources))
        if overlapping_sources:
            new_row = {
                "Chromosome": row_i["Chromosome"],
                "Start": start,
                "End": end,
                "source": current_source,
                "labels": ",".join(overlapping_sources),
            }
            new_df_rows.append(new_row)

    return pl.DataFrame(new_df_rows)


def consolidate_csvs(cell_lines_directory: str) -> pl.DataFrame:
    assert os.path.exists(
        cell_lines_directory
    ), f"{cell_lines_directory} does not exist."
    assert os.path.isdir(
        cell_lines_directory
    ), f"{cell_lines_directory} is not a directory."

    cell_line_labels = get_cell_line_labels(cell_lines_directory)

    chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

    all_dfs = []
    for folder in cell_line_labels:
        folder_path = os.path.join(cell_lines_directory, folder, "peaks")

        for file in os.listdir(folder_path):
            if file.endswith("filtered.broadPeak"):
                file_path = os.path.join(folder_path, file)

                # Adjust the column names to meet pyranges' requirements

                df = pl.read_csv(
                    file_path,
                    separator="\t",
                    has_header=False,
                    columns=[0, 1, 2],
                    new_columns=["Chromosome", "Start", "End"],
                )

                # Filter the DataFrame based on the chromosome list
                df = df.filter(pl.col("Chromosome").is_in(chromosomes))

                # Add a new column indicating the source folder
                df = df.with_columns(pl.lit(folder).alias("source"))

                df = df.with_columns(pl.lit(folder).alias("labels"))

                all_dfs.append(df)

    merged_df = pl.concat(all_dfs)

    queries = []

    for chr in chromosomes:
        lazy_chr_df = merged_df.lazy().filter(pl.col("Chromosome") == chr).lazy()
        sorted_df = lazy_chr_df.sort(by=["Chromosome", "Start", "End"]).lazy()
        processed_lazy_chr = sorted_df.map_batches(
            lambda df, current_chr=chr: process_chromosome(current_chr, df)
        )
        queries.append(processed_lazy_chr)

    results = pl.collect_all(queries)

    # Combine results back into a single dataframe
    combined_df = pl.concat(results)

    return combined_df


if __name__ == "__main__":
    num_cores = 32

    positive = consolidate_csvs(
        "/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines"
    )

    print(positive)

    sorted_df = positive.sort(by=["Chromosome", "Start", "End"])

    print(sorted_df)
    positive.write_csv("data/positive_updated.bed", separator="\t", has_header=False)
