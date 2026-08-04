#!/usr/bin/env python3
# WAJ 2026-07-30
#
# Variant of marginalization_naive.py. The ONLY conceptual change: instead of
# fabricating the ATAC background channel from scratch (random / constant / zeros),
# we draw REAL ATAC pileup windows sampled from processed regions. The pileup
# windows are obtained via call_seqlets.process_region_data_fast, which returns an
# `atac_pileup_list` of shape (N, 4096) of raw per-base pileup counts.
#
# Raw pileups are untransformed, but the model was trained on ATAC that is
# log10-transformed and Savitzky-Golay smoothed (see
# src/training/tf_finetuning/tf_dataloader.py: TransformType.LOG10 + FilterType.SAVGOL).
# We reproduce that same normalization here (toggle NORMALIZE_ATAC) so the sampled
# background stays in-distribution for the model.

import os
import pickle
from datetime import datetime

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from scipy.signal import savgol_filter

from src.inference.scripts.deepseq import DeepSeq
from src.analysis.interpretability.call_seqlets import process_region_data_fast
from einops.layers.torch import Rearrange
from tqdm.auto import tqdm

from tangermeme.utils import random_one_hot, _cast_as_tensor, characters
from tangermeme.marginalize import marginalize
from tangermeme.io import read_meme

from matplotlib import pyplot as plt
import seaborn

seaborn.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

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
    Maps the old state dictionary to work with our new model architecture that uses unique activations.

    This function creates a new state dictionary that maintains all the original weights and parameters
    while being compatible with our new unique activation function structure.

    Args:
        old_state_dict (OrderedDict): The original model's state dictionary

    Returns:
        OrderedDict: A new state dictionary compatible with our modified architecture
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


def load_model(model_dir: str, num_tfs=1):
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
        os.path.join(model_dir, "best_model.pth"), map_location=device, weights_only=True
    )

    intermediate_state_dict = map_state_dict_to_unique_activations(state_dict)

    modified_state_dict = {
        key.replace("_orig_mod.", ""): value
        for key, value in intermediate_state_dict.items()
    }

    model.load_state_dict(modified_state_dict)

    model.to(device)

    return model


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_dir, num_tfs=1)
    return model.to(device)


def normalize_atac_pileups(pileups, window_length=5, polyorder=2):
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
    logged = np.where(pileups > 0, np.log10(pileups, where=pileups > 0), 0.0)
    # Savitzky-Golay smoothing along the sequence axis (same defaults as SAVGOL).
    window_length = window_length if window_length % 2 == 1 else window_length + 1
    smoothed = savgol_filter(logged, window_length=window_length,
                             polyorder=polyorder, axis=-1)
    return smoothed


def load_atac_pileup_pool(parquet_path, n_pool, normalize=True, drop_empty=True):
    """
    Build a pool of real ATAC pileup windows to sample the background from.

    Reads the first `n_pool` regions of a processed parquet and runs them through
    call_seqlets.process_region_data_fast, whose third return value is the raw
    ATAC pileup list of shape (n_pool, 4096).

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
            print(f"Dropping {n_dropped} all-zero pileup windows (no accessibility).")
        atac_pileup_list = atac_pileup_list[keep]

    if atac_pileup_list.shape[0] == 0:
        raise RuntimeError(
            "No usable ATAC pileup windows were loaded; check parquet_path / cell "
            "line pileup directories."
        )

    if normalize:
        atac_pileup_list = normalize_atac_pileups(atac_pileup_list)

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

    plt.imshow(y_delta[:, idxs].T, aspect='auto', cmap='RdBu_r', vmin=-ymax, vmax=ymax, interpolation='nearest')
    plt.colorbar(label="Difference in Logits")
    plt.xlabel("Sequence")
    plt.yticks([])
    plt.grid(False)
    plt.savefig(out_fname, dpi='figure')
    plt.close()


class ChannelsLastWrapper(torch.nn.Module):
    """tangermeme feeds (B, C, L); DeepSeq expects (B, L, C)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X, *args, **kwargs):
        return self.model(X.transpose(1, 2), *args, **kwargs)



today = datetime.now().strftime("%Y%m%d")

root_dir = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"
out_dir = os.path.join(root_dir, "marginalization_output")
os.makedirs(out_dir, exist_ok=True)
motif_fn = os.path.join(root_dir,
                        "data/jaspar_motif_memes/20260710_all.meme")
model_dir = os.path.join(root_dir, "data/models")

# --- Real-ATAC background config -------------------------------------------------
# Processed parquet to source real ATAC pileup windows from. Any processed parquet
# with chr_name/start/end/cell_line/attributions works; swap as needed.
atac_source_parquet = os.path.join(root_dir, "data/processed_results/AR_A-375_processed.parquet")
N_PILEUP_POOL = 2048     # regions to read into the sampling pool
NORMALIZE_ATAC = True    # apply the model's log10 + SavGol ATAC preprocessing
# --------------------------------------------------------------------------------

model = model_fn(model_dir)
wrapped = ChannelsLastWrapper(model)

n_samples = 512

raw_motifs = read_meme(motif_fn)
print("="*80)
motifs = dict()
for name, ohe in raw_motifs.items():
    tf = name.split(".")[-1]
    print(tf)
    motif_str = characters(pwm_consensus(ohe, thresh=0.5), alphabet=['A', 'C', 'G', 'T', 'N'])
    print(motif_str)
    print("="*80)
    motifs[tf] = motif_str

atac_methods = ['random', 'keep', 'zeros', 'ones']
starts = ['center', 819, 3277]

# Build the background from REAL ATAC pileup windows instead of from scratch.
atac_pool = load_atac_pileup_pool(atac_source_parquet, n_pool=N_PILEUP_POOL,
                                  normalize=NORMALIZE_ATAC)
X = synthetic_data_existingATAC(atac_pool, n_samples=n_samples)
X.shape


results = dict()
for TF in tqdm(motifs.keys(), desc="Motif"):
    motif = motifs[TF]
    atac_results = dict()
    for atac_method in tqdm(atac_methods, desc="Marginalization", leave=False):
        motif_position_results = dict()
        for motif_position in starts:
            print("="*90)
            print(f"Using {TF} motif: {motif}")
            print()
            print("Using parameters:")
            print(f"  ATAC METHOD: {atac_method}")
            print(f"  START: {motif_position}")
            print(f"  n SAMPLES: {n_samples}")
            print("="*90)

            out_fname = os.path.join(out_dir, f"{today}_{TF}_existingATAC_{atac_method}_{motif_position}.png")


            if isinstance(motif_position, str):
                if motif_position.lower() == "center":
                    start = None
            elif isinstance(motif_position, int):
                start = motif_position
            else:
                raise ValueError(f"Invalid motif_position parameter ({motif_position}, {type(motif_position)}). Must be an integer < (4096 - motif length) OR 'center'.")
            y_before, y_after = marginalize(wrapped, X, motif, start=start, device=device, atac_method=atac_method, verbose=False)
            y_delta = y_after - y_before
            motif_position_results[motif_position] = y_delta
            plot_margins(y_delta, out_fname=out_fname, ylim=1.5)
        atac_results[atac_method] = motif_position_results
    results[TF] = atac_results

with open(os.path.join(out_dir, "jaspar_all_results_existingATAC.pkl"), "wb") as f:
    pickle.dump(results, f)
