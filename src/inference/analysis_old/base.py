import os
import polars as pl
import tempfile
import subprocess


# notebook_dir = os.path.dirname(os.path.abspath("__file__"))
# sys.path.append(notebook_dir)
# from src.inference.aws_inference import process_jsonl_files


def run_bedtools_command(command: str) -> None:
    subprocess.run(command, shell=True, check=True)

project_path = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"
model = "AR"
sample = "22Rv1"
# ground_truth_file = "/data1/datasets_1/human_prostate_PDX/processed/external_data/ChIP_atlas/AR/SRX8406456.05.bed"
ground_truth_file = "/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/22Rv1/bam/22Rv1_merge.sorted.nodup.shifted.bam"

df = pl.read_parquet(project_path + "/data/processed_results/" + model + "_" + sample + "_processed.parquet")
df = df.rename({"chr_name": "chr"})

print(df.head())


def intersect_bed_files(main_df: pl.DataFrame, intersect_df: pl.DataFrame, region_type: str = None) -> pl.DataFrame:
    """
    Intersect two BED files using bedtools and return the original DataFrame with overlap flags.
    
    Args:
        main_df: Primary Polars DataFrame with BED data
        intersect_df: Secondary Polars DataFrame to intersect with
        region_type: Optional region type label to add to results
        
    Returns:
        Original DataFrame with additional column indicating overlaps
    """

    temp_dir = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/tmp"
    with tempfile.NamedTemporaryFile(delete=False, mode='w', dir=temp_dir) as main_file, \
         tempfile.NamedTemporaryFile(delete=False, mode='w', dir=temp_dir) as intersect_file, \
         tempfile.NamedTemporaryFile(delete=False, mode='w', dir=temp_dir) as result_file:
        
        main_path = main_file.name
        intersect_path = intersect_file.name
        result_path = result_file.name

        # Write DataFrames to temporary files
        main_df.write_csv(main_path, separator="\t", include_header=False)
        intersect_df.write_csv(intersect_path, separator="\t", include_header=False)

        # Run bedtools intersect with -c flag to count overlaps
        command = f"bedtools intersect -a {main_path} -b {intersect_path} -c > {result_path}"
        run_bedtools_command(command)

        # Read results back into Polars DataFrame
        result_df = pl.read_csv(
            result_path,
            separator="\t",
            has_header=False,
            new_columns=[*main_df.columns, "overlap_count"]
        )

    # Clean up temporary files
    os.remove(main_path)
    os.remove(intersect_path) 
    os.remove(result_path)

    # Add boolean overlap column
    result_df = result_df.with_columns(
        pl.col("overlap_count").gt(0).alias("overlaps_ground_truth")
    ).drop("overlap_count")

    return result_df

df_ground_truth = pl.read_csv(ground_truth_file, 
                             separator="\t", 
                             has_header=False,
                             new_columns=["chr", "start", "end"],
                             columns=[0,1,2])

intersected_df = intersect_bed_files(df[["chr", "start", "end"]], df_ground_truth)

threshold = 0.99

# save file to cell_line_name_model_threshold.bed
intersected_df.write_csv(sample + "_" + model + "_" + str(threshold) + ".bed", separator="\t", include_header=False)

# get how many 1s in targets
print("Number of CHIP hits in ATAC peaks", df["targets"].sum())
# get ATAC peaks with probability >= threshold
df_positive = df.filter(pl.col("probabilities") >= threshold)
# get number of 1s in targets
print("Number of CHIP hits in ATAC peaks with probability >= threshold:", df_positive["targets"].sum())
# get length of df_positive
print("Number of ATAC peaks with probability >= threshold:", len(df_positive))
# get ground truth positives
df_ground_truth_positive = df.filter(pl.col("targets") == 1)
print("Number of CHIP hits in ground truth:", len(df_ground_truth_positive))
# get ground truth negatives
df_ground_truth_negative = df.filter(pl.col("targets") == 0)
print("Number of Negatives in ground truth:", len(df_ground_truth_negative))


# calculate precision, recall, f1 score
precision = df_positive["targets"].sum() / len(df_positive)
recall = df_ground_truth_positive["targets"].sum() / len(df_ground_truth_positive)
f1_score = 2 * precision * recall / (precision + recall)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)





# add overlaps ground truth to df from intersected_df
df = df.join(intersected_df, on=["chr", "start", "end"], how="left")
# add overlaps_ground_truth to df under targets, 1 if overlaps_ground_truth is true, 0 otherwise
df = df.with_columns(pl.when(pl.col("overlaps_ground_truth")).then(1).otherwise(0).alias("targets"))