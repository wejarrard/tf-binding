#!/usr/bin/env python3
# WAJ 2026-07-30
"""
corr3_atac_pileup_corrs.py
==========================

Sibling to corr2: instead of asking whether the ATAC *attribution* channel co-varies with the
sequence-attribution channels, we ask whether the model's ATAC **attribution** tracks the
**actual raw ATAC pileup signal**.  In other words: where the model assigns importance to
chromatin accessibility, is there really open chromatin there?

The comparison is done the same two complementary ways as corr2, over the length-4096 per-position
vectors of each window:

  (A) REGIONAL per-window correlation  [primary]
      Slice each window into equal-width position bins, recompute the per-window correlation
      within each bin, and average across windows -> a "mean correlation vs position" profile.

  (B) PER-POSITION cross-window correlation  [complementary]
      For each of the 4096 offsets, correlate the channel values ACROSS windows -> a length-4096
      correlation profile.

Both channels come from ONE call to `process_region_data_fast` (call_seqlets.py) on the upstream
processed parquet.  That returns, for the same rows in the same order:
    atac_attribution_list  (n, 4096)  -- the model's ATAC-channel attribution
    atac_pileup_list       (n, 4096)  -- raw ATAC pileup fetched live from per-cell-line tabix files
Both are built with identical symmetric padding (adj_start = center - 2048), so the two vectors are
aligned by construction -- no coordinate join is needed.

We compute this for two high-confidence prediction subsets:
    high-confidence POSITIVES (probabilities >= 0.99)
    high-confidence NEGATIVES (probabilities <= 0.01)

---
Usage:
    python3 corr3_atac_pileup_corrs.py                         # AR / VCAP, both subsets
    python3 corr3_atac_pileup_corrs.py --subset pos
    python3 corr3_atac_pileup_corrs.py --subset neg --max-regions 20000
    python3 corr3_atac_pileup_corrs.py --tf AR --cell-line VCAP --bin-size 16 --center-bp 200

Cell line is spelled as in the parquet filename (e.g. VCAP, LNCAP, 22Rv1).
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import polars as pl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------- #
# Path setup.  This file lives at
#   <repo>/src/analysis/interpretability/attr_posthoc_analyses/corr3_atac_pileup_corrs.py
# so parents[1] is the interpretability dir (output root, like corr2) and parents[4] is the repo
# root, which must be on sys.path for `from src.analysis...` (and call_seqlets' own `from src.utils`
# import) to resolve.
# --------------------------------------------------------------------- #
ROOT_DIR = str(Path(__file__).resolve().parents[1])       # .../src/analysis/interpretability
REPO_ROOT = str(Path(__file__).resolve().parents[4])      # repo root (contains src/)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# The processed parquets live in the MAIN checkout, not the worktree (see call_seqlets.project_path).
PROJECT_PATH = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"

DEFAULT_TF = "AR"
DEFAULT_CELL_LINE = "VCAP"
# The VCAP processed parquet uses the "AR-CellLine" prefix; other cell lines differ, so pass
# --parquet explicitly when running anything other than AR/VCAP.
DEFAULT_PARQUET = os.path.join(
    PROJECT_PATH, "data/processed_results/AR-CellLine_VCAP_processed.parquet"
)

# Reuse the correlation math from corr2 (same-dir import; works when run as a script). Importing
# call_seqlets runs its harmless module-level config block (sets model/sample, prints a path) but
# reads no data.
from corr2_finegrain_corrs import (          # noqa: E402
    make_bins,
    rowwise_correlations,
    positional_corrs,
    smooth,
)
from src.analysis.interpretability.call_seqlets import process_region_data_fast  # noqa: E402


# --------------------------------------------------------------------- #
def get_data(parquet_path, subset, base_pileup_dir=None, max_regions=None):
    """Load one probability subset and return (X=atac_attrs, Y=atac_pileup), both (n, 4096).

    Uses process_region_data_fast so the ATAC attribution channel and the raw ATAC pileup come
    from the same rows on the same 4096-bp coordinate frame.
    """
    print("=" * 90)
    print(f"Reading '{subset}' subset from {parquet_path}")
    print("=" * 90)

    lf = pl.scan_parquet(parquet_path)
    if subset == "pos":
        lf = lf.filter(pl.col("probabilities") >= 0.99)
    elif subset == "neg":
        lf = lf.filter(pl.col("probabilities") <= 0.01)
    else:
        raise ValueError(f"subset must be 'pos' or 'neg', got {subset!r}")

    lf = lf.select(
        pl.col("chr_name").alias("chr"),
        "start",
        "end",
        "cell_line",
        "attributions",
        "probabilities",
    )
    if max_regions is not None:
        lf = lf.head(max_regions)

    df = lf.collect()
    n = len(df)
    if max_regions is not None and n >= max_regions:
        print(f"WARNING: --max-regions={max_regions} reached; results cover only the first "
              f"{n} matching windows, not the full '{subset}' subset.")
    if n == 0:
        raise SystemExit(f"No rows matched the '{subset}' subset in {parquet_path}")

    base = Path(base_pileup_dir) if base_pileup_dir is not None else None
    _, atac_attribution_list, atac_pileup_list = process_region_data_fast(df, base_pileup_dir=base)

    X = np.asarray(atac_attribution_list, dtype=np.float32)
    Y = np.asarray(atac_pileup_list, dtype=np.float32)
    if X.shape != Y.shape:
        raise ValueError(f"Matrices are not of same shape: {X.shape} != {Y.shape}")
    return X, Y


def plot_all(prof, pos_r, center_bp, out_dir,
             subset="pos",
             bin_size=20,
             width=4096,
             figsize=(16, 9)):
    """Two-panel figure: (A) regional binned profile, (B) per-position cross-window profile."""
    print("=" * 90)
    print("Plotting...")
    print("=" * 90)

    half_c = center_bp / 2.0
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # (A) Regional per-window correlations
    ax = axes[0]
    off = prof["center_offset"].to_numpy()
    ax.plot(off, prof["pearson_mean"].to_numpy(), "-o", ms=3, color="#3b75af", label="Pearson")
    ax.plot(off, prof["spearman_mean"].to_numpy(), "-o", ms=3, color="#c1622f", label="Spearman")
    ax.axvspan(-half_c, half_c, color="gold", alpha=0.18, label=f"center {center_bp} bp")
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.set_title(f"(A) regional per-window correlation  [{subset}]")
    ax.set_xlabel("position offset from window center (bp)")
    ax.set_ylabel("mean corr(atac_attrs, atac_pileup)")
    ax.legend(fontsize=8)

    # (B) Per-position cross-window correlations
    ax = axes[1]
    pos_off = np.arange(width) - width / 2.0
    ax.plot(pos_off, pos_r, color="0.6", lw=0.4, label="raw")
    ax.plot(pos_off, smooth(pos_r, 32), color="#2f7d3b", lw=1.6, label="smoothed (32 bp)")
    ax.axvspan(-half_c, half_c, color="gold", alpha=0.18, label=f"center {center_bp} bp")
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.set_title(f"(B) per-position cross-window correlation  [{subset}]")
    ax.set_xlabel("position offset from window center (bp)")
    ax.set_ylabel("corr(atac_attrs, atac_pileup) across windows")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"bin{bin_size}_{subset}_pileup_corrs.png"), dpi=300)
    plt.close(fig)


def run_subset(subset, args, out_dir, tag):
    X, Y = get_data(args.parquet, subset,
                    base_pileup_dir=args.base_pileup_dir,
                    max_regions=args.max_regions)
    n, width = X.shape
    bins = make_bins(width, args.bin_size)
    print(f"[{subset}] windows: {n} | window: {width} bp | bins: {len(bins)} x {args.bin_size} bp | "
          f"center region: {args.center_bp} bp\n")

    bin_corrs, bin_spearmans = rowwise_correlations(X, Y, bins)
    prof = pl.DataFrame({
        "bin": list(range(len(bins))),
        "lo": [b[0] for b in bins],
        "center_offset": [b[2] for b in bins],
        "pearson_mean": list(bin_corrs.values()),
        "spearman_mean": list(bin_spearmans.values()),
    })
    prof.write_csv(os.path.join(out_dir, f"{tag}_{subset}_rowwise_correlations.csv"))

    pos_r = positional_corrs(X, Y)
    pl.DataFrame({"offset": np.arange(width) - width // 2, "pearson": pos_r}).write_csv(
        os.path.join(out_dir, f"{tag}_{subset}_positional_profile.csv")
    )

    plot_all(prof, pos_r,
             center_bp=args.center_bp,
             out_dir=out_dir,
             subset=subset,
             bin_size=args.bin_size,
             width=width,
             figsize=(16, 9))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--rootdir", default=ROOT_DIR)
    parser.add_argument("--tf", default=DEFAULT_TF)
    parser.add_argument("--cell-line", default=DEFAULT_CELL_LINE,
                        help="Cell line, parquet spelling (default VCAP)")
    parser.add_argument("--parquet", default=DEFAULT_PARQUET,
                        help="Processed parquet with chr_name/start/end/cell_line/attributions/"
                             "probabilities (default: AR-CellLine_VCAP_processed.parquet)")
    parser.add_argument("--subset", default="both", choices=["pos", "neg", "both"],
                        help="Prediction subset: pos (prob>=0.99), neg (prob<=0.01), or both")
    parser.add_argument("--bin-size", type=int, default=20)
    parser.add_argument("--center-bp", type=int, default=400)
    parser.add_argument("--max-regions", type=int, default=None,
                        help="Cap regions per subset (bounds tabix-fetch runtime; warns when hit)")
    parser.add_argument("--base-pileup-dir", default=None,
                        help="Override root of per-cell-line pileup dirs "
                             "(default: merged_cell_lines/ inside process_region_data_fast)")
    args = parser.parse_args()

    if not os.path.exists(args.parquet):
        raise SystemExit(f"Input parquet not found: {args.parquet}")

    tag = f"{args.tf}_{args.cell_line}"
    out_dir = os.path.join(args.rootdir, f"attr_analyses_output/{tag}/atac_pileup_corr")
    os.makedirs(out_dir, exist_ok=True)

    subsets = ["pos", "neg"] if args.subset == "both" else [args.subset]
    for subset in subsets:
        run_subset(subset, args, out_dir, tag)

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
