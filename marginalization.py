#!/usr/bin/env python3
# WAJ 2026-07-30

import os
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Tuple
from pathlib import Path
import pysam
import json

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn

from scipy.signal import savgol_filter

from src.inference.scripts.deepseq import DeepSeq
from einops.layers.torch import Rearrange
from tqdm.auto import tqdm

from tangermeme.utils import random_one_hot, _cast_as_tensor, characters
from tangermeme.marginalize import marginalize
from tangermeme.io import read_meme

from matplotlib import pyplot as plt
import seaborn

seaborn.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300

# ------------------------------------------------------------------------------


def process_pileups_batch(pileup_dir: Path,
                          regions_df: pl.DataFrame) -> Dict[int, np.ndarray]:
    """Read pileups for many regions of one cell line, opening each tabix file once."""
    context_length = 4_096
    chromosomes = regions_df['chr'].unique().to_list()
    chr_tabix_files = {}

    print(f"Opening tabix files for {len(chromosomes)} chromosomes...")
    for chr_name in tqdm(chromosomes, desc="Loading chromosome files", leave=False):
        pileup_file = pileup_dir / f"{chr_name}.pileup.gz"
        if pileup_file.exists():
            chr_tabix_files[chr_name] = pysam.TabixFile(str(pileup_file))

    atac_arrays = {}
    total_regions = len(regions_df)
    for row in tqdm(regions_df.iter_rows(named=True), total=total_regions,
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
                    valid_mask = (array_indices >= 0) & (
                        array_indices < context_length)
                    atac_array[array_indices[valid_mask]] = counts[valid_mask]
            except Exception as e:
                print(
                    f"Warning: Could not fetch data for {chr_name}:{adj_start}-{adj_end}: {e}")

        atac_arrays[idx] = atac_array

    print("Closing tabix files...")
    for tabixfile in chr_tabix_files.values():
        tabixfile.close()

    return atac_arrays


def reshape_attributions_fast(df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized attribution reshape -> (DNA attrs n,4,4096), (ATAC attrs n,4096)."""
    print("Reshaping attribution data...")
    attributions = np.array(df['attributions'].to_list())
    reshaped = attributions.reshape(-1, 4096, 5)
    attrs_list = reshaped[..., :4].transpose(0, 2, 1)
    atac_attribution_list = reshaped[..., 4]
    return attrs_list, atac_attribution_list


def process_region_data_fast(
        df: pl.DataFrame,
        base_pileup_dir: Path = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape attributions and pull raw ATAC pileups in one pass over cell-line groups."""
    if base_pileup_dir is None:
        base_pileup_dir = Path(
            "/home/ec2-user/projects/tf-binding/data/ATAC_pileups")

    print(f"Processing {len(df)} regions across cell lines...")
    attrs_list, atac_attribution_list = reshape_attributions_fast(df)

    df_with_idx = df.with_row_index("idx")
    atac_pileup_arrays = [None] * len(df)
    cell_line_groups = list(df_with_idx.group_by("cell_line"))

    print(f"Processing {len(cell_line_groups)} cell lines...")
    for cell_line, group_df in tqdm(cell_line_groups, desc="Processing cell lines"):
        cell_line_name = cell_line[0]
        pileup_dir = base_pileup_dir / cell_line_name

        if not pileup_dir.exists():
            print(f"Warning: Pileup directory does not exist: {pileup_dir}")
            for row in group_df.iter_rows(named=True):
                atac_pileup_arrays[row['idx']] = np.zeros(4096)
            continue

        print(
            f"Processing {len(group_df)} regions for cell line: {cell_line_name}")
        atac_arrays_dict = process_pileups_batch(pileup_dir, group_df)
        for idx, atac_array in atac_arrays_dict.items():
            atac_pileup_arrays[idx] = atac_array

    print("Converting to final numpy arrays...")
    atac_pileup_list = np.array(atac_pileup_arrays)
    print("Processing complete!")
    return attrs_list, atac_attribution_list, atac_pileup_list


def pwm_consensus(X, thresh=0.5):
    """
    Take in a PWM and return the consensus.

    This function will take in a PWM that encodes the probabilities of each
    character at each position and will return the consensus, which is the
    most likely individual sequence according to that PWM. This is done by
    taking the argmax at each position. When multiple characters at the same
    position have the maximum probability, the character in the earlier
    numerical position is chosen. If a column has a sum equal to 0 (no
    characters allowed there) the returned consensus also has entirely 0s.


    Parameters
    ----------
    X: torch.Tensor, numpy.ndarray, shape=(alphabet_len, pwm_len)
            A PWM containing the probabilities of each character at each position.


    Returns
    -------
    Y: torch.Tensor, shape=(alphabet_len, pwm_len)
            A PWM of the same shape as `X` except with the maximum-value character
            set to 1.
    """

    X = _cast_as_tensor(X)

    alpha_idxs = X.argmax(dim=0)
    N_idxs = X.amax(dim=0) < thresh
    N_idxs = N_idxs.nonzero().squeeze()

    Y = torch.zeros_like(X)
    Y = torch.cat((Y, torch.zeros((1, Y.shape[-1]))))

    Y[alpha_idxs, torch.arange(X.shape[-1])] = 1
    Y[:, X.sum(dim=0) == 0] = 0

    Y[4, N_idxs] = 1
    Y[:4, N_idxs] = 0
    return Y


def map_state_dict_to_unique_activations(old_state_dict):
    """
    Maps the old state dictionary to work with our new model architecture that 
    uses unique activations.

    This function creates a new state dictionary that maintains all the 
    original weights and parameters while being compatible with our new 
    unique activation function structure.

    Args:
        old_state_dict (OrderedDict): The original model's state dictionary

    Returns:
        OrderedDict: A new state dictionary compatible with our modified 
        architecture
    """
    new_state_dict = {}

    # Create a mapping of old keys to new keys
    key_mapping = {}

    # Iterate through old state dict keys
    for key in old_state_dict.keys():
        # The key structure usually looks like 'layer_name.sublayer.weight'
        new_key = key

        # Check if this is an activation function parameter
        if 'relu' in key.lower() or 'gelu' in key.lower():
            # Extract the layer path to create a unique identifier
            path_parts = key.split('.')
            layer_path = '_'.join(path_parts[:-1])
            new_key = f"{layer_path}.unique_{path_parts[-1]}"

        key_mapping[key] = new_key
        new_state_dict[new_key] = old_state_dict[key]

    return new_state_dict


def load_model(model_dir: str, device, num_tfs=1):
    model = DeepSeq.from_hparams(
        dim=1536,
        depth=11,
        heads=8,
        target_length=-1,
        num_cell_lines=num_tfs,
        return_augs=True,
        num_downsamples=3,
    )

    model.out = nn.Sequential(
        nn.Linear(model.dim * 2, num_tfs),
        Rearrange("... c o -> ... o c"),
        nn.Linear(512, 1),
        nn.Flatten(),
    ).to(device)

    state_dict = torch.load(
        os.path.join(model_dir, "best_model.pth"),
        map_location=device,
        weights_only=True,
    )

    intermediate_state_dict = map_state_dict_to_unique_activations(state_dict)

    modified_state_dict = {
        key.replace("_orig_mod.", ""): value
        for key, value in intermediate_state_dict.items()
    }

    model.load_state_dict(modified_state_dict)

    model.to(device)

    return model


def model_fn(model_dir, device):
    model = load_model(model_dir, device=device, num_tfs=1)
    return model.to(device)


def normalize_atac_pileups(pileups,
                           window_length=5,
                           polyorder=2,
                           method='savgol'):
    """
    Reproduce the ATAC preprocessing the model was trained with.

    The finetuning dataloader (src/training/tf_finetuning/tf_dataloader.py) applies
    TransformType.LOG10 then FilterType.SAVGOL to the raw per-base pileup counts.
    We mirror that here so sampled real pileups are on the same scale the model saw.

    Parameters
    ----------
    pileups: numpy.ndarray, shape=(n, L)
            Raw ATAC pileup windows (per-base counts).

    Returns
    -------
    numpy.ndarray, shape=(n, L)
            log10-transformed + Savitzky-Golay smoothed pileups.
    """
    pileups = np.asarray(pileups, dtype=np.float64)
    # log10(count) with 0 wherever count <= 0, matching apply_transform(LOG10).
    if method.upper() == "BOTH":

        logged = np.where(pileups > 0, np.log10(
            pileups, where=pileups > 0), 0.0)
        # Savitzky-Golay smoothing along the sequence axis (same defaults as SAVGOL).
        window_length = window_length if window_length % 2 == 1 else window_length + 1
        smoothed = savgol_filter(logged, window_length=window_length,
                                 polyorder=polyorder, axis=-1)
        return smoothed
    elif "LOG" in method.upper():
        return np.where(pileups > 0, np.log10(pileups, where=pileups > 0), 0.0)
    elif method.upper() in {"SAVGOL", "SMOOTHING", "SMOOTH", "SMOOTHED"}:
        window_length = window_length if window_length % 2 == 1 else window_length + 1
        smoothed = savgol_filter(pileups, window_length=window_length,
                                 polyorder=polyorder, axis=-1)
        return smoothed


def load_atac_pileup_pool(parquet_path,
                          n_pool,
                          normalize: str = None,
                          drop_empty=True):
    """
    Build a pool of real ATAC pileup windows to sample the background from.

    Reads the first `n_pool` regions of a processed parquet and runs them 
    through call_seqlets.process_region_data_fast, whose third return value is
    the raw ATAC pileup list of shape (n_pool, 4096).

    Parameters
    ----------
    parquet_path: str
            Processed parquet containing chr_name/start/end/cell_line/attributions.
    n_pool: int
            Number of regions (rows) to read into the pool. Kept modest because the
            'attributions' column is large; a few thousand windows is plenty to
            sample a background from.
    normalize: bool
            If True, apply the model's log10 + SavGol ATAC preprocessing.
    drop_empty: bool
            If True, discard all-zero pileup windows (e.g. regions whose cell line
            had no pileup directory) so the background reflects real accessibility.

    Returns
    -------
    numpy.ndarray, shape=(<=n_pool, 4096)
            Pool of ATAC pileup windows.
    """
    print(f"Loading up to {n_pool} regions from: {parquet_path}")
    df = pl.read_parquet(
        parquet_path,
        columns=["chr_name", "start", "end", "cell_line", "attributions"],
        n_rows=n_pool,
    ).rename({"chr_name": "chr"})

    # process_region_data_fast returns (attrs_list, atac_attribution_list, atac_pileup_list)
    _, _, atac_pileup_list = process_region_data_fast(df)
    atac_pileup_list = np.asarray(atac_pileup_list, dtype=np.float64)

    if drop_empty:
        keep = atac_pileup_list.any(axis=1)
        n_dropped = int((~keep).sum())
        if n_dropped:
            print(
                f"Dropping {n_dropped} all-zero pileup windows (no accessibility).")
        atac_pileup_list = atac_pileup_list[keep]

    if atac_pileup_list.shape[0] == 0:
        raise RuntimeError(
            "No usable ATAC pileup windows were loaded; check parquet_path / cell "
            "line pileup directories."
        )

    if normalize:
        atac_pileup_list = normalize_atac_pileups(
            atac_pileup_list, method=normalize)

    print(f"ATAC pileup pool ready: {atac_pileup_list.shape}")
    return atac_pileup_list


def synthetic_data_existingATAC(atac_pool, n_samples: int = 100, seed: int = 0):
    """
    Build marginalization inputs whose ATAC channel is a real pileup window.

    Same as marginalization_naive.synthetic_data_naive except the ATAC channel is
    NOT fabricated: for each of the `n_samples` random DNA backbones we draw a real
    ATAC pileup window (with replacement) from `atac_pool`.

    Parameters
    ----------
    atac_pool: numpy.ndarray, shape=(pool_size, 4096)
            Pool of real ATAC pileup windows (see load_atac_pileup_pool).
    n_samples: int
            Number of background sequences to build.
    seed: int
            Seed for the window sampling (DNA backbones stay random_one_hot).

    Returns
    -------
    torch.Tensor, shape=(n_samples, 5, 4096)
            Channels: 4 one-hot DNA + 1 ATAC.
    """
    length = atac_pool.shape[-1]
    synthetic_seq = random_one_hot((n_samples, 4, length)).type(torch.float32)

    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, atac_pool.shape[0], size=n_samples)
    sampled = atac_pool[idxs]  # (n_samples, length)
    synthetic_atac = torch.from_numpy(sampled).type(torch.float32).unsqueeze(1)

    X = torch.cat((synthetic_seq, synthetic_atac), dim=1)
    return X


def plot_margins(y_delta, out_fname, ylim=None):
    idxs = torch.flip(torch.argsort(y_delta.mean(axis=0)), dims=(0,))
    ymax = max(abs(y_delta.min()), abs(y_delta.max())) if not ylim else ylim

    plt.imshow(y_delta[:, idxs].T,
               aspect='auto',
               cmap='RdBu_r',
               vmin=-ymax,
               vmax=ymax,
               interpolation='nearest')
    plt.colorbar(label="Difference in Logits")
    plt.xlabel("Sequence")
    plt.yticks([])
    plt.grid(False)
    plt.savefig(out_fname, dpi='figure')
    plt.close()


def to_long(obj, value_name="value"):
    """One row per (motif, atac_method, position, sample)."""
    recs = []
    for motif, by_method in obj.items():
        for method, by_pos in by_method.items():
            for pos, arr in by_pos.items():
                a = np.asarray(arr).reshape(-1)          # (64, 1) -> (64,)
                recs.append(
                    pd.DataFrame({
                        "motif": motif,
                        "atac_method": method,
                        "position": pos,
                        "sample": np.arange(a.size),
                        value_name: a,
                    })
                )
    df = pd.concat(recs, ignore_index=True)
    df["motif"] = df["motif"].astype("category")
    df["atac_method"] = df["atac_method"].astype("category")
    df["position"] = df["position"].astype("int32")
    df["sample"] = df["sample"].astype("int32")
    return df.sort_values(["motif", "atac_method", "position", "sample"],
                          ignore_index=True)


class ChannelsLastWrapper(torch.nn.Module):
    """tangermeme feeds (B, C, L); DeepSeq expects (B, L, C)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X, *args, **kwargs):
        return self.model(X.transpose(1, 2), *args, **kwargs)


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("="*80)
    print(f"Using device: {device}")
    print("="*80)

    TFs_of_interest = {
        "AR",
        "FOXA1",
        "FOXA2",
        "ERG",
        # "EGF2",
        # "EGF1",
        "MYC",
        "FOS::JUN",
        "JUN",
        "JUNB",
        "JUND",
        "NR3C1",
        "POU5F1",
        "CTCF",
        "ASCL1",
        "HOXB13",
    }

    # =============================== CONFIG =============================== #
    transcription_factor = "FOXA1"
    cell_line = "22Rv1"

    save_timestamp = False  # print timestamp in output directory?

    # --- Spike-In config -----------------------------------------------------
    n_samples = 2 ** 5
    seq_step = 2 ** 4
    include_zero_spikein = True
    limit_motifs = True
    # -------------------------------------------------------------------------

    # --- Real-ATAC background config -----------------------------------------
    # Processed parquet to source real ATAC pileup windows from. Any processed parquet
    # with chr_name/start/end/cell_line/attributions works; swap as needed.
    N_PILEUP_POOL = 2048            # regions to read into the sampling pool
    # apply the model's log10 and/or SavGol ATAC preprocessing ['log' | 'savgol' | 'both' | None]
    NORMALIZE_ATAC = "both"
    atac_methods = [                # any of [ 'keep' | 'zeros' | 'ones' | 'random' | 'shuffle' ]
        'keep',
        # 'ones',
        # 'zeros',
        # 'shuffle',
        # 'random',
    ]
    # -------------------------------------------------------------------------

    # ======================================================================= #

    time_zone = ZoneInfo("America/Los_Angeles")
    tag = f"{transcription_factor}_{cell_line}"

    root_dir = "/home/ec2-user/projects/tf-binding"
    os.makedirs(os.path.join(
        root_dir, "marginalization_output"), exist_ok=True)
    model_dir = os.path.join(root_dir, f"data/models/{transcription_factor}")
    motif_path = os.path.join(
        root_dir, "data/jaspar_motif_memes/JASPAR_CORE_human_latest.meme")

    ts = datetime.now(tz=time_zone).strftime("%Y%m%d:%H%M")
    if save_timestamp:
        out_dir = os.path.join(root_dir, f"marginalization_output/{tag}_{ts}")
    else:
        out_dir = os.path.join(root_dir, f"marginalization_output/{tag}")

    os.makedirs(out_dir, exist_ok=True)


    model = model_fn(model_dir, device=device)
    wrapped = ChannelsLastWrapper(model)

    raw_motifs = read_meme(motif_path)
    motifs = dict()
    for name, ohe in raw_motifs.items():
        tf = name.split(" ")[-1]
        # tf = name
        # print(tf)
        motif_str = characters(pwm_consensus(ohe, thresh=0.5), alphabet=[
                               'A', 'C', 'G', 'T', 'N'])
        # print(motif_str)
        # print("="*80)
        motifs[tf] = motif_str

    if limit_motifs:
        motifs = {k: v for k, v in motifs.items() if k.upper()
                  in TFs_of_interest}
        not_found = TFs_of_interest - set(motifs.keys())
        if len(not_found) > 0:
            print("="*90)
            print(f"[WARNING] The following TFs were not found in the motif file: {not_found}")
            print("="*90)

    for k, v in motifs.items():
        print("="*80)
        print(k)
        print(v)
    print("="*80)

    config = {
        "transcription_factor": transcription_factor,
        "cell_line": cell_line,
        "limit_motifs": limit_motifs,
        "TFs_of_interest": list(TFs_of_interest),
        "not_found": list(not_found) if limit_motifs else "NA",
        "n_samples": n_samples,
        "seq_step": seq_step,
        "include_zero_spikein": include_zero_spikein,
        "motif_path": motif_path,
        "N_PILEUP_POOL": N_PILEUP_POOL,
        "NORMALIZE_ATAC": NORMALIZE_ATAC,
        "atac_methods": atac_methods,
        "model_dir": model_dir,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print("="*80)
    print(f"Analyzing {len(motifs)} motifs.")
    print("="*80)

    # starts = range(0, 4001, 25)
    if include_zero_spikein:
        starts = [i for i in range(0, 4001, seq_step)]
    else:
        starts = [i for i in range(0, 4001, seq_step)][1:]

    atac_source_parquet = os.path.join(
        root_dir, f"data/attr_parquets/{tag}_processed.parquet")
    # Build the background from REAL ATAC pileup windows instead of from scratch.
    atac_pool = load_atac_pileup_pool(atac_source_parquet, n_pool=N_PILEUP_POOL,
                                      normalize=NORMALIZE_ATAC)
    X = synthetic_data_existingATAC(atac_pool, n_samples=n_samples)
    X.shape

    results = dict()
    for TF in tqdm(motifs.keys(), desc="Motifs"):
        motif = motifs[TF]
        atac_results = dict()
        for atac_method in tqdm(atac_methods, 
                                desc="ATAC Methods", 
                                leave=False):
            motif_position_results = dict()
            for motif_position in tqdm(starts, 
                                       desc="Positions", 
                                       leave=False):
                tqdm.write("="*90)
                tqdm.write(f"Using {TF} motif: {motif}")
                tqdm.write("")
                tqdm.write("Using parameters:")
                tqdm.write(f"  ATAC METHOD: {atac_method}")
                tqdm.write(f"  START: {motif_position}")
                tqdm.write(f"  n SAMPLES: {n_samples}")

                if isinstance(motif_position, str):
                    if motif_position.lower() == "center":
                        start = None
                elif isinstance(motif_position, int):
                    start = motif_position
                else:
                    raise ValueError(
                        f"Invalid motif_position parameter ({motif_position}, {type(motif_position)}). Must be an integer < (4096 - motif length) OR 'center'.")
                y_before, y_after = marginalize(
                    wrapped, X, motif, start=start, device=device, atac_method=atac_method, verbose=False)
                y_delta = y_after - y_before
                motif_position_results[motif_position] = y_delta
                tqdm.write("Finished! Statistics:")
                tqdm.write(f"  Mean delta:\n    {y_delta.mean():.4f}")
                tqdm.write(f"  Median delta:\n    {y_delta.median():.4f}")
                tqdm.write("="*90)
            atac_results[atac_method] = motif_position_results

        results[TF] = atac_results

    df = to_long(results)
    df.to_csv(os.path.join(out_dir, "marginalization_values.csv"), index=False)


if __name__ == "__main__":
    main()
