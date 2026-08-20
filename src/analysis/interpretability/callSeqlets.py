#!/usr/bin/env python
# WAJ 2026-08-18

import argparse
import tempfile
import json
import os
import sys

import pysam

import torch
import pandas as pd
import polars as pl
import numpy as np

from tqdm.auto import tqdm
from datetime import datetime
from typing import Dict, Tuple
from pathlib import Path
from tangermeme.seqlet import recursive_seqlets, tfmodisco_seqlets

from src.utils.generate_training_peaks import run_bedtools_command


def wrangle_attributions(attributions):
    attrs = np.array(attributions.to_list())
    attrs = attrs.reshape(-1, 4096, 5)
    atac_attrs = attrs[..., 4]
    attrs = attrs[..., :4]
    neg_attrs = np.clip(attrs, None, 0)
    pos_attrs = np.clip(attrs, 0, None)
    pos_sums = np.array([mat.sum(axis=1) for mat in pos_attrs])
    neg_sums = np.array([mat.sum(axis=1) for mat in neg_attrs])
    return pos_sums, neg_sums, atac_attrs


def mutate_df_attrs(df: pl.LazyFrame):
    df = df.filter(
        (pl.col("probabilities") <= 0.01) | (pl.col("probabilities") >= 0.99)
    ).collect(engine="streaming")

    pos_sums, neg_sums, atac_attrs = wrangle_attributions(df.get_column('attributions'))

    df = df.with_columns(pl.Series("pos_attrs", pos_sums.tolist(), dtype=pl.List(pl.Float64)))
    df = df.with_columns(pl.Series("neg_attrs", neg_sums.tolist(), dtype=pl.List(pl.Float64)))
    df = df.with_columns(pl.Series("atac_attrs", atac_attrs.tolist(), dtype=pl.List(pl.Float64)))

    return df


def write_attr_matrix(parquet_file, output_path, compression:str="zstd"):
    print("Loading in parquet file...")
    df = pl.read_parquet(
        parquet_file,
        columns=['chr_name', 'start', 'end', 'targets', 'predicted', 'probabilities', 'attributions'],
        parallel="columns",
        use_statistics=True,
        use_pyarrow=True,
        memory_map=True,
    ).lazy()

    print("Modifying dataframe...")
    df = mutate_df_attrs(df)

    df.write_parquet(output_path,
                     compression=compression,
                     row_group_size=100_000)

def intersect_bed_files(main_df: pl.LazyFrame,
                        intersect_df: pl.DataFrame,
                        region_type: str = None) -> pl.LazyFrame:
    """Intersect two BED frames via bedtools and tag overlap on the main frame."""
    _schema = main_df.collect_schema()
    main_cols = _schema.keys()

    with tempfile.NamedTemporaryFile(delete=False, mode='w') as main_file, \
            tempfile.NamedTemporaryFile(delete=False, mode='w') as intersect_file, \
            tempfile.NamedTemporaryFile(delete=False, mode='w') as result_file:
        main_path = main_file.name
        intersect_path = intersect_file.name
        result_path = result_file.name

        main_df.collect().write_csv(main_path, separator="\t", include_header=False)
        intersect_df.write_csv(
            intersect_path,
            separator="\t",
            include_header=False)

        run_bedtools_command(
            f"bedtools intersect -a {main_path} -b {intersect_path} -c > {result_path}")

        result_df = pl.read_csv(
            result_path,
            separator="\t",
            has_header=False,
            new_columns=[*main_cols, "overlap_count"],
        ).lazy()

        os.remove(main_path)
        os.remove(intersect_path)
        os.remove(result_path)

        return result_df.with_columns(
            pl.col("overlap_count").gt(0).alias("overlaps_ground_truth")
        ).drop("overlap_count")
    

def reshape_attributions_fast(
        df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized attribution reshape -> (DNA attrs n,4,4096), (ATAC attrs n,4096)."""
    attributions = np.array(df['attributions'].to_list())
    reshaped = attributions.reshape(-1, 4096, 5)
    attrs_list = reshaped[..., :4].transpose(0, 2, 1)
    atac_attribution_list = reshaped[..., 4]
    return attrs_list, atac_attribution_list


def process_pileups_batch(pileup_dir: Path,
                          regions_df: pl.DataFrame) -> Dict[int, np.ndarray]:
    """Read pileups for many regions of one cell line, opening each tabix file once."""
    context_length = 4_096
    chromosomes = regions_df['chr'].unique().to_list()
    chr_tabix_files = {}

    print(f"Opening tabix files for {len(chromosomes)} chromosomes...")
    for chr_name in tqdm(
            chromosomes,
            desc="Loading chromosome files",
            leave=False):
        pileup_file = pileup_dir / f"{chr_name}.pileup.gz"
        if pileup_file.exists():
            chr_tabix_files[chr_name] = pysam.TabixFile(str(pileup_file))

    atac_arrays = {}
    total_regions = len(regions_df)
    for row in tqdm(regions_df.iter_rows(named=True),
                    total=total_regions,
                    desc="Processing regions", leave=False):
        idx, chr_name, start, end = row['idx'], row['chr'], row['start'], row['end']

        interval_length = end - start
        extra_seq = context_length - interval_length
        extra_left_seq = extra_seq // 2
        extra_right_seq = extra_seq - extra_left_seq
        adj_start = start - extra_left_seq
        adj_end = end + extra_right_seq

        atac_array = np.zeros(context_length)

        if chr_name in chr_tabix_files:
            tabixfile = chr_tabix_files[chr_name]
            positions, counts = [], []
            try:
                for rec in tabixfile.fetch(chr_name, adj_start, adj_end):
                    fields = rec.split("\t")
                    positions.append(int(fields[1]))
                    counts.append(float(fields[3]))

                if positions:
                    positions = np.array(positions)
                    counts = np.array(counts)
                    array_indices = positions - adj_start
                    valid_mask = (
                        array_indices >= 0) & (
                        array_indices < context_length)
                    atac_array[array_indices[valid_mask]] = counts[valid_mask]
            except Exception as e:
                tqdm.write(
                    f"Warning: Could not fetch data for {chr_name}:{adj_start}-{adj_end}: {e}")

        atac_arrays[idx] = atac_array

    print("Closing tabix files...")
    for tabixfile in chr_tabix_files.values():
        tabixfile.close()

    return atac_arrays


def process_region_data_fast(df: pl.DataFrame,
                             base_pileup_dir: Path) -> Tuple[np.ndarray,
                                                             np.ndarray,
                                                             np.ndarray]:
    """Reshape attributions and pull raw ATAC pileups in one pass over cell-line groups."""
    print(f"Processing {len(df)} regions across cell lines...")
    attrs_list, atac_attribution_list = reshape_attributions_fast(df)

    df_with_idx = df.with_row_index("idx")
    atac_pileup_arrays = [None] * len(df)
    cell_line_groups = list(df_with_idx.group_by("cell_line"))

    for cell_line, group_df in cell_line_groups:
        cell_line_name = cell_line[0]
        pileup_dir = base_pileup_dir / cell_line_name

        if not pileup_dir.exists():
            print(f"Warning: Pileup directory does not exist: {pileup_dir}")
            for row in group_df.iter_rows(named=True):
                atac_pileup_arrays[row['idx']] = np.zeros(4096)
            continue

        print(
            f"Processing {
                len(group_df)} regions for cell line: {cell_line_name}")
        atac_arrays_dict = process_pileups_batch(pileup_dir, group_df)
        for idx, atac_array in atac_arrays_dict.items():
            atac_pileup_arrays[idx] = atac_array

    print("Converting to final numpy arrays...")
    atac_pileup_list = np.array(atac_pileup_arrays)
    print("Processing complete!")
    return attrs_list, atac_attribution_list, atac_pileup_list


def get_seqlets(attrs_list,
                method='recursive',
                **kwargs):
    attrs_array = np.stack(attrs_list, axis=0)
    summed_attrs = attrs_array.sum(axis=1)

    if method == 'tfmodisco':
        summed_attrs_tensor = torch.from_numpy(summed_attrs).float()
        seqlets = tfmodisco_seqlets(summed_attrs_tensor)
    elif method == 'recursive':
        r_kwargs = {'threshold': 0.01, 'min_seqlet_len': 4,
                    'max_seqlet_len': 25, 'additional_flanks': 0}
        r_kwargs.update(kwargs)
        seqlets = recursive_seqlets(summed_attrs, **r_kwargs)
        seqlets = seqlets[seqlets['attribution'] > 0]
    else:
        raise ValueError(f"Unknown seqlet calling method: '{method}'.")

    nt_idx = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    sequences = []
    for i in range(len(seqlets)):
        s = seqlets.iloc[i]
        start, end, sample_idx = int(
            s['start']), int(
            s['end']), int(
            s['example_idx'])
        sample_attrs = attrs_array[sample_idx, :, start:end].T.squeeze()
        hits = np.argmax(sample_attrs, axis=1)
        sequences.append(''.join(nt_idx[i] for i in hits))
    seqlets['sequence'] = sequences
    return seqlets


def add_genomic_coords(seqlets: pd.DataFrame,
                       ground_truth_df: pl.DataFrame,
                       context_length: int = 4096) -> pd.DataFrame:
    """Add absolute genomic coords (chr, g_start, g_end) to seqlets.

    The seqlet 'start'/'end' are relative to the 4096bp model window, and
    'example_idx' is the row position in `ground_truth_df` -- which MUST be the
    same in-memory frame that built the attribution arrays this run, since the
    streaming collect order in build_ground_truth_df is not reproducible across
    runs. Mirrors the symmetric padding used to build the window (see
    process_pileups_batch): adj_start = p_start - (context_length - L) // 2.
    """
    chrom = ground_truth_df["chr"].to_numpy()
    g_s = ground_truth_df["start"].to_numpy()
    g_e = ground_truth_df["end"].to_numpy()
    ex = seqlets["example_idx"].to_numpy().astype(int)
    adj_start = g_s[ex] - (context_length - (g_e[ex] - g_s[ex])) // 2
    seqlets = seqlets.copy()
    seqlets["chr"] = chrom[ex]
    seqlets["g_start"] = adj_start + seqlets["start"].to_numpy().astype(int)
    seqlets["g_end"] = adj_start + seqlets["end"].to_numpy().astype(int)
    return seqlets


def write_fasta(seqlets, outfile):
    sequences = seqlets['sequence'].tolist()

    chr = seqlets['chr'].tolist()
    g_start = [str(i) for i in seqlets['g_start'].tolist()]
    fa_headers = [f">{'_'.join(i)}" for i in zip(chr, g_start)]
    g_end = [str(i) for i in seqlets['g_end'].tolist()]
    fa_headers = ["-".join(i) for i in zip(fa_headers, g_end)]
    fasta_dict = dict(zip(fa_headers, sequences))
    with open(outfile, 'w') as f:
        for header, seq in fasta_dict.items():
            f.write(f'{header}\n{seq}\n')



def main():
    parser = argparse.ArgumentParser(
        description="Seqlet calling and analysis pipeline.")
    parser.add_argument("--tf", type=str, default="AR",
                        help="transcription factor of interest (e.g., AR, CTCF).")
    parser.add_argument("--cell_line", type=str, default="22Rv1",
                        help="cell line name (e.g. VCAP, 22Rv1).")
    parser.add_argument("--use_tfmodisco", action="store_true")
    parser.add_argument("--gt_subset", type=str, default="TP",
                        help="TP | TN | FP | FN")
    parser.add_argument("--gt_count_threshold", type=int, default=2)
    parser.add_argument("--mtx", action='store_true',
                        help="Write Attribution matrices to file? Requires lots of (> 128G+) RAM.")
    args = parser.parse_args()

    transcription_factor = args.tf
    cell_line = args.cell_line

    seqlet_method = "tfmodisco" if args.use_tfmodisco else "recursive"

    ground_truth_subset = args.gt_subset
    ground_truth_count_threshold = args.gt_count_threshold

    root_dir = "/home/ec2-user/projects/tf-binding/"
    interpretability_dir = os.path.join(root_dir, "src/analysis/interpretability")


    today = datetime.now().strftime("%Y%m%d")
    tag = f"{transcription_factor}_{cell_line}"

    seqlet_export_dir = os.path.join(
        interpretability_dir,
        f"seqlets/{transcription_factor}_{cell_line}",
    )
    os.makedirs(seqlet_export_dir, exist_ok=True)
    base_pileup_dir = os.path.join(root_dir, "data/ATAC_pileups")

    gt_bed_path = Path(f"{root_dir}/data/ground_truth_beds")

    if args.mtx:
        attr_matrix_dir = os.path.join(interpretability_dir, "attribution_matrices")
        os.makedirs(attr_matrix_dir, exist_ok=True)
        attr_matrix_path = os.path.join(attr_matrix_dir, f"{tag}.parquet")

    seqlets_path = os.path.join(
        seqlet_export_dir, f"seqlets_{ground_truth_subset}.csv"
    )

    parquet_file = Path(f"{root_dir}/data/attr_parquets/{transcription_factor}_{cell_line}_processed.parquet")
    with open(
        "/home/ec2-user/projects/tf-binding/src/analysis/interpretability/parquet_bed_map.json",
        "r") as f:
        parquet_bed_map = json.load(f)

    if parquet_file.name in parquet_bed_map.keys():
        ground_truth_file = f"{gt_bed_path}/{parquet_bed_map[parquet_file.name]}"
    else:
        print("ERROR: no ground-truth bed file to match parquet.")
        sys.exit()


    config_path = os.path.join(seqlet_export_dir, f"config.json")
    config = {
        "date": today,
        "transcription_factor": transcription_factor,
        "cell_line": cell_line,
        "seqlet_method": seqlet_method,
        "ground_truth_subset": ground_truth_subset,
        "ground_truth_count_threshold": ground_truth_count_threshold,
        "paths": {
            "root_dir": root_dir,
            "interpretability_dir": interpretability_dir,
            "parquet_file": str(parquet_file),
            "ground_truth_file": ground_truth_file,
            "seqlet_export_dir": seqlet_export_dir,
            "attr_matrix_dir": attr_matrix_dir if args.mtx else "NA",
        },
    }
    with open(config_path, "w") as _f:
        json.dump(config, _f, indent=2)

    if args.mtx:
        write_attr_matrix(
            parquet_file, 
            output_path = attr_matrix_path)

    df = pl.read_parquet(
        parquet_file,
        columns=["chr_name", "start", "end", "cell_line",
                "targets", "predicted", "weights",
                "probabilities", "attributions"],
        parallel="columns",
        use_statistics=True,
        memory_map=True,
    ).lazy()
    df = df.rename({"chr_name": "chr"})


    df_ground_truth = pl.read_csv(
        ground_truth_file,
        separator="\t",
        has_header=False,
        new_columns=["chr", "start", "end", "count"],
        columns=[0, 1, 2, 3],
    )
    df_ground_truth = df_ground_truth.filter(
        pl.col("count") >= ground_truth_count_threshold)


    intersected_df = intersect_bed_files(df.select(["chr", "start", "end"]),
                                        df_ground_truth)


    df = df.join(intersected_df, on=["chr", "start", "end"],
                how="left")

    df = df.with_columns(
        pl.when(pl.col("overlaps_ground_truth"))
        .then(1)
        .otherwise(0)
        .alias("targets"))


    # Label entries for GT
    df = df.with_columns(correct=pl.when((pl.col("targets") == 1) & (pl.col("predicted") == 1))
                         .then(pl.lit("TP"))
                         .when((pl.col("targets") == 0) & (pl.col("predicted") == 0))
                         .then(pl.lit("TN"))
                         .when((pl.col("targets") == 1) & (pl.col("predicted") == 0))
                         .then(pl.lit("FN"))
                         .when((pl.col("targets") == 0) & (pl.col("predicted") == 1))
                         .then(pl.lit("FP"))).collect(engine="streaming")

    ground_truth_df = df.filter(
        pl.col("correct") == ground_truth_subset)

    attrs, _, _ = process_region_data_fast(
        ground_truth_df,
        base_pileup_dir=Path(base_pileup_dir),
    )

    seqlets = get_seqlets(attrs, method='recursive')
    seqlets = add_genomic_coords(seqlets, ground_truth_df)


    seqlets.to_csv(seqlets_path, index=False)
    write_fasta(seqlets, os.path.join(seqlet_export_dir, "seqlets.fa"))


if __name__ == "__main__":
    main()