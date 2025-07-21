# %%
import os
import sys
import numpy as np
import polars as pl
import torch

notebook_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(notebook_dir)



project_path = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"
model = "AR"
sample = "22Rv1"
jaspar_file = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/analysis/interpretability/motifs/AR.jaspar"  # Update this path
ground_truth_file = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/transcription_factors/AR/merged/22RV1_AR_merged.bed"

df = pl.read_parquet("/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/processed_results/AR_22Rv1_processed.parquet", 
                    columns=["chr_name", "start", "end", "cell_line", "targets", "predicted", "weights", "probabilities", "attributions"],
                    parallel="columns",                     # Enable parallel reading
                    use_statistics=True,                    # Use parquet statistics
                    memory_map=True).lazy()                         # Use memory mapping
df = df.rename({"chr_name": "chr"})



# %%
import os
import tempfile
import polars as pl
from src.utils.generate_training_peaks import run_bedtools_command

def intersect_bed_files(main_df: pl.LazyFrame, intersect_df: pl.DataFrame, region_type: str = None) -> pl.LazyFrame:
    """
    Intersect two BED files using bedtools and return the original DataFrame with overlap flags.
    Args:
    main_df: Primary Polars DataFrame with BED data
    intersect_df: Secondary Polars DataFrame to intersect with
    region_type: Optional region type label to add to results
    Returns:
    Original DataFrame with additional column indicating overlaps
    """
    # Get column names from schema
    main_cols = main_df.schema.keys()
    
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as main_file, \
         tempfile.NamedTemporaryFile(delete=False, mode='w') as intersect_file, \
         tempfile.NamedTemporaryFile(delete=False, mode='w') as result_file:
        main_path = main_file.name
        intersect_path = intersect_file.name
        result_path = result_file.name
        
        # Write DataFrames to temporary files - collect LazyFrame first
        main_df.collect().write_csv(main_path, separator="\t", include_header=False)
        intersect_df.write_csv(intersect_path, separator="\t", include_header=False)
        
        # Run bedtools intersect with -c flag to count overlaps
        command = f"bedtools intersect -a {main_path} -b {intersect_path} -c > {result_path}"
        run_bedtools_command(command)
        
        # Read results back into Polars DataFrame
        result_df = pl.read_csv(
            result_path,
            separator="\t",
            has_header=False,
            new_columns=[*main_cols, "overlap_count"]
        ).lazy()
        
        # Clean up temporary files
        os.remove(main_path)
        os.remove(intersect_path)
        os.remove(result_path)
        
        # Add boolean overlap column
        return result_df.with_columns(
            pl.col("overlap_count").gt(0).alias("overlaps_ground_truth")
        ).drop("overlap_count")

HIGH_COUNT_QUANTILE = 0.75
MAX_COUNT_THRESHOLD = 30
MID_COUNT_THRESHOLD = 10

def threshold_peaks(df):
    """
    Filter peaks based on count thresholds.
    Works with both DataFrame and LazyFrame.
    """
    # Handle scalar operations safely
    def get_scalar(expr):
        if isinstance(df, pl.LazyFrame):
            return expr.collect().item()
        return expr.item()
    
    max_count = get_scalar(df.select(pl.col("count").max()))
    
    if max_count <= 2:
        return df
    elif max_count > MAX_COUNT_THRESHOLD:
        threshold = get_scalar(df.select(pl.col("count").quantile(HIGH_COUNT_QUANTILE)))
        return df.filter(pl.col("count") > threshold)
    elif max_count > MID_COUNT_THRESHOLD:
        threshold = get_scalar(df.select(pl.col("count").median()))
        return df.filter(pl.col("count") > threshold)
    
    return df

# Usage example:
df_ground_truth = pl.read_csv(ground_truth_file,
                             separator="\t",
                             has_header=False,
                             new_columns=["chr", "start", "end", "count"],
                             columns=[0,1,2,3])

df_ground_truth_filtered = threshold_peaks(df_ground_truth)

# Use select() instead of subscripting
intersected_df = intersect_bed_files(df.select(["chr", "start", "end"]), df_ground_truth_filtered)

# add overlaps ground truth to df from intersected_df
ground_truth_df = df.join(intersected_df, on=["chr", "start", "end"], how="left")

# add overlaps_ground_truth to df under targets, 1 if overlaps_ground_truth is true, 0 otherwise
ground_truth_df = ground_truth_df.with_columns(
    pl.when(pl.col("overlaps_ground_truth")).then(1).otherwise(0).alias("targets")
)

# %%
# Step 1: Keep the filtering lazy until collection
# Corrected: Added parentheses around individual conditions
df_positive_correct = (
    ground_truth_df
        .filter( (pl.col("targets") == 1) & (pl.col("predicted") == 1) )
        .collect(streaming=True)               # materialise to DataFrame
        # .sample(n=10_000, seed=42, with_replacement=False)  # down-sample
)

# df_negative_correct_all = ground_truth_df.filter(
#     (pl.col("targets") == 0) & (pl.col("predicted") == 0)
# ).collect()

# # Step 2: Get the count of positive samples
# pos_count = len(df_positive_correct)

# # Step 3: Sample from the materialized negative DataFrame
# df_negative = df_negative_correct_all.sample(
#     n=min(pos_count, len(df_negative_correct_all)), seed=42
# )

# # Step 4: Concatenate the two DataFrames
# df_balanced = pl.concat([df_positive_correct, df_negative])

# df_balanced

# %%
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List, Dict
import pysam

def process_pileups(pileup_dir: Path, chr_name: str, start: int, end: int) -> pl.DataFrame:
    """Process pileup files for a given genomic region with 4096bp context."""
    context_length = 4_096
    interval_length = end - start
    extra_seq = context_length - interval_length
    extra_left_seq = extra_seq // 2
    extra_right_seq = extra_seq - extra_left_seq
    start -= extra_left_seq
    end += extra_right_seq
    
    # Get the pileup file for the given chromosome
    pileup_file = pileup_dir / f"{chr_name}.pileup.gz"
    assert pileup_file.exists(), f"pileup file for {pileup_file} does not exist"
    
    tabixfile = pysam.TabixFile(str(pileup_file))
    records = []
    for rec in tabixfile.fetch(chr_name, start, end):
        records.append(rec.split("\t"))
    
    # Convert records to a DataFrame using Polars
    df = pl.DataFrame({
        "chr_name": [rec[0] for rec in records],
        "position": [int(rec[1]) for rec in records],
        "nucleotide": [rec[2] for rec in records],
        "count": [float(rec[3]) for rec in records],
    })
    
    return df


def create_position_to_count_mapping(pileup_df: pl.DataFrame) -> Dict[int, float]:
    """Create a mapping from genomic position to ATAC count."""
    return dict(zip(pileup_df['position'].to_list(), pileup_df['count'].to_list()))


def create_atac_pileup_array(position_count_map: Dict[int, float], 
                            start_pos: int, 
                            length: int = 4096) -> np.ndarray:
    """Create ATAC pileup array for a genomic region."""
    atac_array = np.zeros(length)
    
    for i in range(length):
        pos = start_pos + i
        if pos in position_count_map:
            atac_array[i] = position_count_map[pos]
    
    return atac_array


def reshape_attributions_fast(df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast reshape attribution data using vectorized operations.
    
    Returns:
        attrs_list: Attribution scores for ACGT (shape: n_samples, 4, 4096)
        atac_attribution_list: ATAC attribution scores (shape: n_samples, 4096)
    """
    print("Reshaping attribution data...")
    # Convert to numpy array more efficiently
    attributions = np.array(df['attributions'].to_list())
    
    # Vectorized reshape - much faster than loops
    reshaped = attributions.reshape(-1, 4096, 5)
    
    # Split into ACGT and ATAC components
    attrs_list = reshaped[..., :4].transpose(0, 2, 1)  # Shape: (n_samples, 4, 4096)
    atac_attribution_list = reshaped[..., 4]  # Shape: (n_samples, 4096)
            
    return attrs_list, atac_attribution_list


def process_pileups_batch(pileup_dir: Path, regions_df: pl.DataFrame) -> Dict[int, np.ndarray]:
    """Process multiple pileup regions for a single cell line efficiently."""
    context_length = 4_096
    atac_arrays = {}
    
    # Get unique chromosomes to minimize file operations
    chromosomes = regions_df['chr'].unique().to_list()
    chr_tabix_files = {}
    
    # Open all needed tabix files once
    print(f"Opening tabix files for {len(chromosomes)} chromosomes...")
    for chr_name in tqdm(chromosomes, desc="Loading chromosome files", leave=False):
        pileup_file = pileup_dir / f"{chr_name}.pileup.gz"
        if pileup_file.exists():
            chr_tabix_files[chr_name] = pysam.TabixFile(str(pileup_file))
    
    # Process each region
    region_iterator = regions_df.iter_rows(named=True)
    total_regions = len(regions_df)
    
    for row in tqdm(region_iterator, total=total_regions, desc="Processing regions", leave=False):
        idx, chr_name, start, end = row['idx'], row['chr'], row['start'], row['end']
        
        # Calculate adjusted coordinates
        interval_length = end - start
        extra_seq = context_length - interval_length
        extra_left_seq = extra_seq // 2
        extra_right_seq = extra_seq - extra_left_seq
        adj_start = start - extra_left_seq
        adj_end = end + extra_right_seq
        
        # Initialize array
        atac_array = np.zeros(context_length)
        
        # Get data if tabix file exists
        if chr_name in chr_tabix_files:
            tabixfile = chr_tabix_files[chr_name]
            
            # Collect all positions and counts at once
            positions = []
            counts = []
            
            try:
                for rec in tabixfile.fetch(chr_name, adj_start, adj_end):
                    fields = rec.split("\t")
                    positions.append(int(fields[1]))
                    counts.append(float(fields[3]))
                
                # Vectorized assignment
                if positions:
                    positions = np.array(positions)
                    counts = np.array(counts)
                    
                    # Calculate array indices
                    array_indices = positions - adj_start
                    
                    # Filter valid indices
                    valid_mask = (array_indices >= 0) & (array_indices < context_length)
                    valid_indices = array_indices[valid_mask]
                    valid_counts = counts[valid_mask]
                    
                    # Assign values
                    atac_array[valid_indices] = valid_counts
                    
            except Exception as e:
                print(f"Warning: Could not fetch data for {chr_name}:{adj_start}-{adj_end}: {e}")
        
        atac_arrays[idx] = atac_array
    
    # Close tabix files
    print("Closing tabix files...")
    for tabixfile in chr_tabix_files.values():
        tabixfile.close()
    
    return atac_arrays


def process_region_data_fast(df: pl.DataFrame, base_pileup_dir: Path = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast process both attribution and pileup data using Polars optimizations.
    
    Args:
        df: DataFrame containing attribution data and region info (chr, start, end, cell_line columns)
        base_pileup_dir: Base directory path for pileup files (optional, uses default if None)
    
    Returns:
        attrs_list: Attribution scores for ACGT
        atac_attribution_list: ATAC attribution scores
        atac_pileup_list: Raw ATAC pileup counts
    """
    if base_pileup_dir is None:
        base_pileup_dir = Path("/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/")
    
    print(f"Processing {len(df)} regions across cell lines...")
    
    # Get attribution data (fast vectorized version)
    attrs_list, atac_attribution_list = reshape_attributions_fast(df)
    
    # Add row index for tracking
    df_with_idx = df.with_row_index("idx")
    
    # Group by cell line for batch processing
    atac_pileup_arrays = [None] * len(df)
    cell_line_groups = list(df_with_idx.group_by("cell_line"))
    
    print(f"Processing {len(cell_line_groups)} cell lines...")
    
    for cell_line, group_df in tqdm(cell_line_groups, desc="Processing cell lines"):
        cell_line_name = cell_line[0]
        
        # Construct cell-line specific pileup directory
        pileup_dir = base_pileup_dir / cell_line_name / "pileup_mod"
        
        if not pileup_dir.exists():
            print(f"Warning: Pileup directory does not exist: {pileup_dir}")
            # Fill with zeros for this cell line
            for row in group_df.iter_rows(named=True):
                atac_pileup_arrays[row['idx']] = np.zeros(4096)
            continue
        
        print(f"Processing {len(group_df)} regions for cell line: {cell_line_name}")
        
        # Process all regions for this cell line at once
        atac_arrays_dict = process_pileups_batch(pileup_dir, group_df)
        
        # Assign to the correct positions in the final array
        for idx, atac_array in atac_arrays_dict.items():
            atac_pileup_arrays[idx] = atac_array
    
    print("Converting to final numpy arrays...")
    # Convert to numpy array
    atac_pileup_list = np.array(atac_pileup_arrays)
    
    print("Processing complete!")
    return attrs_list, atac_attribution_list, atac_pileup_list

attrs_list, atac_attribution_list, atac_pileup_list = process_region_data_fast(df_positive_correct)

# %%
# Import additional required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('whitegrid')
from tangermeme.plot import plot_logo
from tangermeme.seqlet import recursive_seqlets, tfmodisco_seqlets
from matplotlib.colors import TwoSlopeNorm



def get_seqlets(attrs_list, use_absolute_values=False, method='recursive', direction='positive', **kwargs):
    """
    Extract seqlets from attribution data using one of two methods.

    Args:
        attrs_list (list): List of attribution arrays for each sample.
        use_absolute_values (bool): Whether to use absolute attribution values for peak finding.
        method (str): The seqlet calling method to use. Either 'tfmodisco' (default)
                      or 'recursive'.
        direction (str): 'positive' to find seqlets in high-attribution regions,
                         'negative' to find them in low-attribution (negative) regions.
        **kwargs: Method-specific arguments.
            For 'tfmodisco':
                - See tangermeme.seqlet.tfmodisco_seqlets documentation. Common
                  parameters include `window_size` and `flank`.
            For 'recursive':
                - threshold (float): p-value threshold for a span to be considered
                                     a seqlet. Default: 0.01.
                - min_seqlet_len (int): Minimum length of a seqlet. Default: 4.
                - max_seqlet_len (int): Maximum length of a seqlet. Default: 25.
                - additional_flanks (int): Number of base pairs to add to each side
                                           of a discovered seqlet. Default: 0.
    """
    attrs_array = np.stack(attrs_list, axis=0)

    # Sum attributions across one-hot encoded dimension to get a per-position score
    summed_attrs = attrs_array.sum(axis=1)

    # If looking for negative contributions, flip the scores
    if direction == 'negative':
        summed_attrs = -summed_attrs
    elif direction != 'positive':
        raise ValueError("direction must be 'positive' or 'negative'")

    # Optionally use absolute values. Note: this will make 'negative' direction meaningless.
    if use_absolute_values:
        summed_attrs = np.abs(summed_attrs)

    if method == 'tfmodisco':
        summed_attrs_tensor = torch.from_numpy(summed_attrs).float()
        seqlets = tfmodisco_seqlets(summed_attrs_tensor)#, **kwargs)
    elif method == 'recursive':
        # # recursive_seqlets uses a p-value based threshold.
        # # We set some reasonable defaults based on its documentation.
        r_kwargs = {
            'threshold': 0.01,
            'min_seqlet_len': 5,
            'max_seqlet_len': 25,
            'additional_flanks': 0
        }
        r_kwargs.update(kwargs)
        seqlets = recursive_seqlets(summed_attrs, **r_kwargs)
    else:
        raise ValueError(f"Unknown seqlet calling method: '{method}'. Choose 'tfmodisco' or 'recursive'.")


    nt_idx = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    # Add sequences to seqlets df
    sequences = []
    for i in range(len(seqlets)):
        sample = seqlets.iloc[i]
        start = int(sample['start'])
        end = int(sample['end'])
        sample_idx = int(sample['example_idx'])

        sample_attrs = attrs_array[sample_idx, :, start:end].T.squeeze()
        hits = np.argmax(sample_attrs, axis=1)
        seq = ''.join([nt_idx[i] for i in hits])
        sequences.append(seq)
    
    seqlets['sequence'] = sequences
    return seqlets
def plot_seqlet_with_atac(seqlets, attrs_list, atac_attribution_list, atac_pileup_list, 
                         sample_rank=0, context_size=20, colormap='RdBu_r', equal_color_scale=False):
    """
    Create a two-panel plot with a NON-SYMMETRIC color-normalized heatmap.
    - Top: DNA base attributions (logo plot)  
    - Bottom: ATAC pileup with attribution heatmap background (0 is always white)
    
    The color scale for the heatmap now stretches to the true min and max of the data in the window.
    """
    # --- This part of the function is unchanged ---
    sample = seqlets.iloc[[sample_rank]]
    slice_idx = int(sample['example_idx'].tolist()[0])
    sequence = sample['sequence'].tolist()[0]
    start = int(sample['start'].tolist()[0])
    end = int(sample['end'].tolist()[0])

    seqlet_center = (start + end) // 2
    seqlet_length = end - start
    total_window_size = seqlet_length + (2 * context_size)
    window_start = seqlet_center - (total_window_size // 2)
    window_end = seqlet_center + (total_window_size // 2)
    window_start = max(0, window_start)
    window_end = min(4096, window_end)
    if window_end - window_start < total_window_size:
        if window_start == 0:
            window_end = min(4096, window_start + total_window_size)
        elif window_end == 4096:
            window_start = max(0, window_end - total_window_size)
    
    print(f"Seqlet: {start}-{end} (center: {seqlet_center})")
    print(f"Window: {window_start}-{window_end} (size: {window_end - window_start})")
    if 'p-value' in sample.columns:
        p_value = sample['p-value'].tolist()[0]
        print(f"P-Value: {p_value}")
    
    plot_coords = np.arange(window_start, window_end)
    X_attr = attrs_list[slice_idx].astype(np.float64)
    atac_attr = atac_attribution_list[slice_idx].astype(np.float64)
    atac_pileup = atac_pileup_list[slice_idx].astype(np.float64)
    X_attr_windowed = X_attr[:, window_start:window_end]
    atac_attr_windowed = atac_attr[window_start:window_end]
    atac_pileup_windowed = atac_pileup[window_start:window_end]
    
    print(f"Windowed shapes: DNA={X_attr_windowed.shape}, ATAC_attr={atac_attr_windowed.shape}, ATAC_pileup={atac_pileup_windowed.shape}")

    fig = plt.figure(figsize=(18, 10), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[20, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.02)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    # Top panel logic remains the same...
    plot_logo(X_attr_windowed, ax=ax1)
    n_ticks = 8
    tick_positions = np.linspace(0, len(plot_coords)-1, n_ticks)
    tick_labels = np.linspace(plot_coords[0], plot_coords[-1], n_ticks).astype(int)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax1.set_xlabel("Genomic Coordinate")
    ax1.set_ylabel("DNA Attributions")
    ax1.set_title(f"DNA Base Attributions | Sample: {slice_idx} | {sequence}")

    # --- This part of the function is also unchanged ---
    heatmap_height = 25
    attr_heatmap = np.tile(atac_attr_windowed, (heatmap_height, 1))
    max_pileup = np.max(atac_pileup_windowed) if len(atac_pileup_windowed) > 0 else 1
    y_max = max_pileup * 1.1

    # <<< MODIFIED SECTION >>>
    # CREATE THE ASYMMETRIC NORMALIZER CENTERED AT 0
    if atac_attr_windowed.size > 0:
        # Get the true min and max of the data in the window
        vmin_val = np.min(atac_attr_windowed)
        vmax_val = np.max(atac_attr_windowed)
    else:
        # Handle empty window case
        vmin_val, vmax_val = -1, 1

    if equal_color_scale:
        # Handle empty window case for np.abs
        if atac_attr_windowed.size > 0:
            vabs_max = np.max(np.abs(atac_attr_windowed))
        else:
            vabs_max = 1
        vmin_val = -vabs_max
        vmax_val = vabs_max 
        
    # Create the normalizer with the actual data bounds, keeping 0 as the center
    norm = TwoSlopeNorm(vcenter=0, vmin=vmin_val, vmax=vmax_val)
    # <<< END OF MODIFIED SECTION >>>

    # Create the heatmap background using the SAME coordinate system and the NEW norm
    im = ax2.imshow(attr_heatmap, 
                    cmap=colormap,
                    aspect='auto',
                    extent=[plot_coords[0], plot_coords[-1], 0, y_max],
                    alpha=0.7,
                    interpolation='bilinear',
                    norm=norm) # Apply the new asymmetric normalizer
    
    # --- The rest of the function is unchanged ---
    ax2.plot(plot_coords, atac_pileup_windowed, color='black', linewidth=2.5, 
             label='ATAC-seq Pileup', alpha=0.9)
    ax2.set_xlim(plot_coords[0], plot_coords[-1])
    
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('ATAC Attribution', rotation=270, labelpad=15, fontsize=11)
    
    ax2.set_xlabel("Genomic Coordinate")
    ax2.set_ylabel("ATAC-seq Signal")
    ax2.set_title(f"ATAC Pileup with Attribution Heatmap | Sample: {slice_idx}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.show()


# %%
seqlets = get_seqlets(attrs_list, method='recursive')

# %%
plot_seqlet_with_atac(seqlets, attrs_list,atac_attribution_list=atac_attribution_list, atac_pileup_list=atac_pileup_list, sample_rank=4, context_size=2000)


# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming your normalization functions are defined as before:
# normalize_rows_minmax, normalize_rows_by_peak, normalize_rows_by_central_peak

def normalize_rows_minmax(arr: np.ndarray) -> np.ndarray:
    """Normalize each row to range [0, 1]."""
    min_vals = arr.min(axis=1, keepdims=True)
    max_vals = arr.max(axis=1, keepdims=True)
    denom = np.clip(max_vals - min_vals, a_min=1e-6, a_max=None)
    return (arr - min_vals) / denom

def normalize_rows_by_peak(arr: np.ndarray) -> np.ndarray:
    """Normalize each row by its maximum value."""
    max_vals = arr.max(axis=1, keepdims=True)
    denom = np.clip(max_vals, a_min=1e-6, a_max=None)
    return arr / denom

def normalize_rows_by_central_peak(arr: np.ndarray, window_size: int = 100) -> np.ndarray:
    """Normalize each row by the maximum value in its central region."""
    center = arr.shape[1] // 2
    half_window = window_size // 2
    # Ensure central_region slicing is robust
    start_slice = max(0, center - half_window)
    end_slice = min(arr.shape[1], center + half_window + (window_size % 2)) # ensure full window size
    central_region = arr[:, start_slice:end_slice]
    
    if central_region.shape[1] == 0: # Handle case where central window is empty (e.g. arr is too small)
        peak_vals = np.ones((arr.shape[0], 1)) * 1e-6 # Avoid division by zero, effectively no scaling
    else:
        peak_vals = central_region.max(axis=1, keepdims=True)
        
    denom = np.clip(peak_vals, a_min=1e-6, a_max=None)
    return arr / denom


def plot_avg_atac_and_attribution(atac_pileup_list: np.ndarray,
                                  atac_attribution_list: np.ndarray,
                                  normalize_pileup_method: str = "minmax",
                                  normalize_attribution_method: str = "peak",
                                  context_size: int = None, # New parameter for zooming
                                  central_peak_norm_window_size: int = 100 # Pass through for central peak norm
                                  ):
    """
    Plot average ATAC pileup and attribution across all samples.
    Normalization methods can be specified.
    Allows zooming to a central context window.
    """
    processed_pileup = atac_pileup_list.copy()
    pileup_norm_label = " (Raw)"
    if normalize_pileup_method == "minmax":
        processed_pileup = normalize_rows_minmax(processed_pileup)
        pileup_norm_label = " (Min-Max Normalized by Sample)"
    elif normalize_pileup_method == "peak":
        processed_pileup = normalize_rows_by_peak(processed_pileup)
        pileup_norm_label = " (Peak Normalized by Sample)"
    elif normalize_pileup_method == "central_peak":
        processed_pileup = normalize_rows_by_central_peak(processed_pileup, window_size=central_peak_norm_window_size)
        pileup_norm_label = " (Central Peak Normalized by Sample)"
    elif normalize_pileup_method != "none":
        raise ValueError(f"Unknown normalize_pileup_method: {normalize_pileup_method}")

    processed_attribution = atac_attribution_list.copy()
    attr_norm_label = " (Raw)"
    if normalize_attribution_method == "minmax":
        processed_attribution = normalize_rows_minmax(processed_attribution)
        attr_norm_label = " (Min-Max Normalized by Sample)"
    elif normalize_attribution_method == "peak":
        processed_attribution = normalize_rows_by_peak(processed_attribution)
        attr_norm_label = " (Peak Normalized by Sample)"
    elif normalize_attribution_method == "central_peak":
        processed_attribution = normalize_rows_by_central_peak(processed_attribution, window_size=central_peak_norm_window_size)
        attr_norm_label = f" (Central Peak[{central_peak_norm_window_size}bp] Normalized by Sample)"
    elif normalize_attribution_method != "none":
        raise ValueError(f"Unknown normalize_attribution_method: {normalize_attribution_method}")

    mean_pileup = processed_pileup.mean(axis=0)
    mean_attr = processed_attribution.mean(axis=0)
    
    full_length = len(mean_pileup)
    x_coords = np.arange(full_length)

    # Determine window for plotting
    if context_size is not None and 0 < context_size < full_length:
        center_idx = full_length // 2
        half_cs = context_size // 2
        
        win_start = max(0, center_idx - half_cs)
        win_end = min(full_length, win_start + context_size)
        
        # Adjust start if win_end hit full_length, to try to maintain context_size
        if win_end == full_length:
            win_start = max(0, full_length - context_size)

        mean_pileup_to_plot = mean_pileup[win_start:win_end]
        mean_attr_to_plot = mean_attr[win_start:win_end]
        x_to_plot = x_coords[win_start:win_end]
        plot_title_suffix = f" (Region: {win_start} - {win_end-1})"
    else:
        mean_pileup_to_plot = mean_pileup
        mean_attr_to_plot = mean_attr
        x_to_plot = x_coords
        plot_title_suffix = " (Full View)"

    if len(x_to_plot) == 0:
        print("Warning: Calculated plot window is empty. Plotting full range instead.")
        mean_pileup_to_plot = mean_pileup
        mean_attr_to_plot = mean_attr
        x_to_plot = x_coords
        plot_title_suffix = " (Full View - Empty Zoom Attempt)"


    fig, ax1 = plt.subplots(figsize=(18, 5), dpi=150)

    ax1.plot(x_to_plot, mean_pileup_to_plot, color='black', label="Avg ATAC Pileup")
    ax1.set_ylabel(f"Pileup{pileup_norm_label}", color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    ax2.plot(x_to_plot, mean_attr_to_plot, color='red', alpha=0.6, label="Avg ATAC Attribution")
    ax2.set_ylabel(f"Attribution{attr_norm_label}", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax1.set_title("Average ATAC Pileup and Attribution" + plot_title_suffix)
    if len(x_to_plot) > 0:
        ax1.set_xlabel(f"Position ({x_to_plot[0]}–{x_to_plot[-1]})")
    else:
        ax1.set_xlabel("Position (Empty Range)")
        
    ax1.grid(True, alpha=0.3)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.tight_layout()
    plt.show()


plot_avg_atac_and_attribution(
    atac_pileup_list, # Make sure this is defined from your data
    atac_attribution_list, # Make sure this is defined from your data
    normalize_pileup_method="none",
    normalize_attribution_method="none",
    central_peak_norm_window_size=100,
    context_size=4000
)


# %%
def plot_average_half_profiles(attribution_data: np.ndarray, data_label: str = "ATAC Track Attribution"):
    num_samples, L = attribution_data.shape
    mid_idx = L // 2

    left_half_data = attribution_data[:, :mid_idx]
    right_half_data = attribution_data[:, mid_idx:]

    avg_left_profile = np.mean(left_half_data, axis=0)
    # Flip the right half for comparison: last position of right half becomes first, etc.
    avg_right_profile_flipped = np.mean(right_half_data[:, ::-1], axis=0) 
    
    x_axis_half = np.arange(mid_idx)

    plt.figure(figsize=(12, 6))
    plt.plot(x_axis_half, avg_left_profile, label='Average Left Half Profile', color='blue', alpha=0.8)
    plt.plot(x_axis_half, avg_right_profile_flipped, label='Average Right Half Profile (Flipped)', color='red', linestyle='--', alpha=0.8)
    
    # Difference plot
    # plt.plot(x_axis_half, avg_left_profile - avg_right_profile_flipped, label='Difference (Left - Flipped Right)', color='green', alpha=0.5)

    plt.xlabel(f"Position from Center (0 to {mid_idx-1})")
    plt.ylabel("Average Attribution Score")
    plt.title(f"Comparison of Average Left vs. Flipped Right {data_label} Profiles")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.show()

# --- Example Usage ---
plot_average_half_profiles(atac_attribution_list, 
                           data_label="ATAC Track Attribution (Positive Correct Sites)")

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Assume 'atac_attribution_list' is your data (e.g., from df_positive_correct or df_balanced)
# It should have shape (number_of_samples, 4096)

def analyze_attribution_asymmetry(attribution_data: np.ndarray, 
                                  data_label: str = "ATAC Track Attribution",
                                  metric: str = 'sum_raw'):
    """
    Analyzes the asymmetry of attribution scores between the left and right halves of sequences.

    Args:
        attribution_data: Numpy array of shape (num_samples, sequence_length).
        data_label: String label for the data being analyzed (for plot titles).
        metric: The metric to use for quantifying attribution:
                'sum_raw': Sum of raw attribution scores.
                'sum_positive': Sum of positive attribution scores.
                'sum_absolute': Sum of absolute attribution scores.
    """
    num_samples, L = attribution_data.shape

    if L % 2 != 0:
        print(f"Warning: Sequence length {L} is odd. The exact center split might be slightly imbalanced.")
    
    # Define split point
    # For L=4096, mid_idx = 2048. Left: 0 to 2047. Right: 2048 to 4095.
    mid_idx = L // 2

    left_half_data = attribution_data[:, :mid_idx]
    right_half_data = attribution_data[:, mid_idx:]

    if metric == 'sum_raw':
        left_values = np.sum(left_half_data, axis=1)
        right_values = np.sum(right_half_data, axis=1)
        metric_label = "Sum of Raw Attributions"
    elif metric == 'sum_positive':
        left_values = np.sum(np.maximum(0, left_half_data), axis=1)
        right_values = np.sum(np.maximum(0, right_half_data), axis=1)
        metric_label = "Sum of Positive Attributions"
    elif metric == 'sum_absolute':
        left_values = np.sum(np.abs(left_half_data), axis=1)
        right_values = np.sum(np.abs(right_half_data), axis=1)
        metric_label = "Sum of Absolute Attributions"
    else:
        raise ValueError(f"Unknown metric: {metric}. Choose from 'sum_raw', 'sum_positive', 'sum_absolute'.")

    asymmetry_scores = left_values - right_values # Positive means left > right

    # --- Statistical Analysis ---
    mean_asymmetry = np.mean(asymmetry_scores)
    median_asymmetry = np.median(asymmetry_scores)
    # Paired t-test (to see if the mean difference is significantly different from 0)
    # Or Wilcoxon signed-rank test if normality is a concern (often more robust for scores)
    # t_stat, p_value = stats.ttest_rel(left_values, right_values)
    wilcoxon_stat, p_value = stats.wilcoxon(left_values, right_values, alternative='two-sided' if mean_asymmetry !=0 else 'greater') # H1: left != right
    
    # --- Plotting Histogram ---
    plt.figure(figsize=(10, 6))
    plt.hist(asymmetry_scores, bins=50, edgecolor='k', alpha=0.7)
    plt.axvline(mean_asymmetry, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_asymmetry:.2f}')
    plt.axvline(median_asymmetry, color='orange', linestyle='dashed', linewidth=2, label=f'Median: {median_asymmetry:.2f}')
    plt.axvline(0, color='black', linestyle='solid', linewidth=1)
    plt.xlabel(f"{metric_label} (Left Half - Right Half)")
    plt.ylabel("Number of Samples")
    plt.title(f"Distribution of {data_label} Asymmetry ({metric_label})\nWilcoxon P-value (Left vs Right): {p_value:.2e}")
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.show()

    print(f"--- Asymmetry Analysis for {data_label} using {metric_label} ---")
    print(f"Number of samples: {num_samples}")
    print(f"Mean Asymmetry (Left - Right): {mean_asymmetry:.4f}")
    print(f"Median Asymmetry (Left - Right): {median_asymmetry:.4f}")
    print(f"Standard Deviation of Asymmetry Scores: {np.std(asymmetry_scores):.4f}")
    print(f"Wilcoxon signed-rank test statistic: {wilcoxon_stat:.4f}, P-value: {p_value:.4g}")

    if p_value < 0.05:
        if mean_asymmetry > 0:
            print("Result: Statistically significant evidence that the Left side has larger attribution.")
        elif mean_asymmetry < 0:
            print("Result: Statistically significant evidence that the Right side has larger attribution.")
        else:
            print("Result: Statistically significant, but mean difference is zero (might indicate symmetric deviations). Check median.")
    else:
        print("Result: No statistically significant difference found between left and right side attributions based on this metric.")
    print("--------------------------------------------------\n")
    return asymmetry_scores, p_value

# --- Example Usage ---
# Make sure 'atac_attribution_list' is loaded and available
# For example, if you processed 'df_positive_correct':
# _, _, atac_attribution_list_positive = process_region_data_fast(df_positive_correct) # Assuming this returns it

# If atac_attribution_list is already the numpy array from your previous code:
analyze_attribution_asymmetry(atac_attribution_list, 
                              data_label="ATAC Track Attribution (Positive Correct Sites)", 
                              metric='sum_raw')

# analyze_attribution_asymmetry(atac_attribution_list, 
#                               data_label="ATAC Track Attribution (Positive Correct Sites)", 
#                               metric='sum_positive')

# analyze_attribution_asymmetry(atac_attribution_list, 
#                               data_label="ATAC Track Attribution (Positive Correct Sites)", 
#                               metric='sum_absolute')

# You can also apply this to your DNA base attributions (attrs_list)
# Assuming attrs_list has shape (num_samples, 4, 4096)
# To analyze DNA base attributions, you might want to sum across the A,C,G,T dimension first
# or analyze each base channel separately if that's meaningful.
# For a general DNA importance, you could sum absolute values across bases, then sum across length:
# dna_importance_per_position = np.sum(np.abs(attrs_list), axis=1) # Shape: (num_samples, 4096)
# analyze_attribution_asymmetry(dna_importance_per_position, 
#                               data_label="Summed Absolute DNA Base Attribution (Positive Correct Sites)", 
#                               metric='sum_raw') # 'sum_raw' here means sum of summed_abs_dna_attr

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats # Make sure scipy is imported for stats.wilcoxon

# Make sure your 'analyze_attribution_asymmetry' and 'plot_average_half_profiles'
# functions from the previous response are defined in your environment.
# (I'm re-pasting them here for completeness in this code block,
#  but you likely already have them)

def analyze_attribution_asymmetry(attribution_data: np.ndarray,
                                  data_label: str = "ATAC Track Attribution",
                                  metric: str = 'sum_raw'):
    """
    Analyzes the asymmetry of attribution scores between the left and right halves of sequences.
    (Same function as provided before)
    """
    num_samples, L = attribution_data.shape
    if L % 2 != 0:
        print(f"Warning: Sequence length {L} is odd. The exact center split might be slightly imbalanced.")
    mid_idx = L // 2
    left_half_data = attribution_data[:, :mid_idx]
    right_half_data = attribution_data[:, mid_idx:]

    if metric == 'sum_raw':
        left_values = np.sum(left_half_data, axis=1)
        right_values = np.sum(right_half_data, axis=1)
        metric_label = "Sum of Raw Attributions"
    elif metric == 'sum_positive':
        left_values = np.sum(np.maximum(0, left_half_data), axis=1)
        right_values = np.sum(np.maximum(0, right_half_data), axis=1)
        metric_label = "Sum of Positive Attributions"
    elif metric == 'sum_absolute':
        left_values = np.sum(np.abs(left_half_data), axis=1)
        right_values = np.sum(np.abs(right_half_data), axis=1)
        metric_label = "Sum of Absolute Attributions"
    else:
        raise ValueError(f"Unknown metric: {metric}. Choose from 'sum_raw', 'sum_positive', 'sum_absolute'.")

    asymmetry_scores = left_values - right_values
    mean_asymmetry = np.mean(asymmetry_scores)
    median_asymmetry = np.median(asymmetry_scores)
    # Use alternative='two-sided' for a general test of difference,
    # or specify 'greater' or 'less' if you have a directional hypothesis beforehand.
    # Given your previous finding, you might expect left > right, so alternative='greater' for left_values > right_values
    # (which means asymmetry_scores > 0).
    # For wilcoxon(x, y, alternative='greater'), it tests if median of x-y is greater than 0.
    # So, wilcoxon(asymmetry_scores, alternative='greater') if testing if median asymmetry > 0
    # or wilcoxon(left_values, right_values, alternative='greater')
    
    # Test if median of (left_values - right_values) is different from 0
    # If testing if left is specifically larger, H1 is median(left-right) > 0
    alt_hypothesis = 'two-sided'
    if np.abs(mean_asymmetry) > 1e-9 : # If there's a clear direction in the mean
        alt_hypothesis = 'greater' if mean_asymmetry > 0 else 'less'

    # For a paired test where we look at the differences:
    # We test if the median of these differences is non-zero.
    # If asymmetry_scores = left - right, we want to test if median(asymmetry_scores) is > 0 if we expect left > right
    if len(asymmetry_scores) > 0 and not np.all(asymmetry_scores == 0): # Wilcoxon needs non-identical samples or non-zero differences
        try:
            # We are testing the 'asymmetry_scores' directly. If we expect left > right, then asymmetry_scores > 0.
            # So, alternative='greater' tests if the median of asymmetry_scores is greater than 0.
            wilcoxon_stat, p_value = stats.wilcoxon(asymmetry_scores, alternative='greater' if mean_asymmetry > 0 else ('less' if mean_asymmetry < 0 else 'two-sided'))

        except ValueError as e: # Can happen if all differences are zero
            print(f"Wilcoxon test could not be performed: {e}")
            wilcoxon_stat, p_value = np.nan, np.nan
    else:
        wilcoxon_stat, p_value = np.nan, np.nan


    plt.figure(figsize=(10, 6))
    plt.hist(asymmetry_scores, bins=50, edgecolor='k', alpha=0.7)
    plt.axvline(mean_asymmetry, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_asymmetry:.2f}')
    plt.axvline(median_asymmetry, color='orange', linestyle='dashed', linewidth=2, label=f'Median: {median_asymmetry:.2f}')
    plt.axvline(0, color='black', linestyle='solid', linewidth=1)
    plt.xlabel(f"{metric_label} (Left Half - Right Half)")
    plt.ylabel("Number of Samples")
    plt.title(f"Distribution of {data_label} Asymmetry ({metric_label})\nWilcoxon P-value (Median Diff > 0 or < 0): {p_value:.2e}")
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.show()

    print(f"--- Asymmetry Analysis for {data_label} using {metric_label} ---")
    print(f"Number of samples: {num_samples}")
    print(f"Mean Asymmetry (Left - Right): {mean_asymmetry:.4f}")
    print(f"Median Asymmetry (Left - Right): {median_asymmetry:.4f}")
    print(f"Standard Deviation of Asymmetry Scores: {np.std(asymmetry_scores):.4f}")
    print(f"Wilcoxon signed-rank test statistic: {wilcoxon_stat:.4f}, P-value: {p_value:.4g}")

    if not np.isnan(p_value) and p_value < 0.05:
        if mean_asymmetry > 0: # And alternative was 'greater'
            print("Result: Statistically significant evidence that the Left side has larger attribution.")
        elif mean_asymmetry < 0: # And alternative was 'less'
             print("Result: Statistically significant evidence that the Right side has larger attribution.")
        else: # This case might occur if alternative was 'two-sided' and p < 0.05 but mean is near 0
             print("Result: Statistically significant difference detected, but mean is close to zero. Inspect median and distribution.")
    else:
        print("Result: No statistically significant difference found (or test not performed) between left and right side attributions based on this metric.")
    print("--------------------------------------------------\n")
    return asymmetry_scores, p_value


def plot_average_half_profiles(attribution_data: np.ndarray, data_label: str = "Generic Attribution"):
    """
    Plots the average profile of the left half vs. the average of the flipped right half.
    (Same function as provided before)
    """
    num_samples, L = attribution_data.shape
    mid_idx = L // 2
    left_half_data = attribution_data[:, :mid_idx]
    right_half_data = attribution_data[:, mid_idx:]
    avg_left_profile = np.mean(left_half_data, axis=0)
    avg_right_profile_flipped = np.mean(right_half_data[:, ::-1], axis=0)
    x_axis_half = np.arange(mid_idx)

    plt.figure(figsize=(12, 6))
    plt.plot(x_axis_half, avg_left_profile, label='Average Left Half Profile', color='blue', alpha=0.8)
    plt.plot(x_axis_half, avg_right_profile_flipped, label='Average Right Half Profile (Flipped)', color='red', linestyle='--', alpha=0.8)
    plt.xlabel(f"Position from Center (0 to {mid_idx-1})")
    plt.ylabel("Average Attribution Score")
    plt.title(f"Comparison of Average Left vs. Flipped Right {data_label} Profiles")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.show()


# --- Applying to DNA Base Attributions ---

# Assume 'attrs_list' is your DNA base attribution data, loaded as a NumPy array
# with shape (number_of_samples, 4, 4096).
# For example, this might come from your 'process_region_data_fast' function
# or directly from where you load your data.

# Example: if attrs_list is loaded:
# attrs_list, _, _ = process_region_data_fast(df_positive_correct) # Make sure this is correctly populated

if 'attrs_list' in locals() and isinstance(attrs_list, np.ndarray) and attrs_list.ndim == 3 and attrs_list.shape[1] == 4:
    print("Processing DNA base attributions (attrs_list)...")
    
    # 1. Preprocess attrs_list to get a per-position DNA importance score.
    # We sum the absolute values of attributions across the 4 bases (A,C,G,T) for each position.
    # This gives a single score per position representing total DNA nucleotide importance.
    # The resulting array will have non-negative values.
    dna_importance_scores_per_position = np.sum(np.abs(attrs_list), axis=1)
    # This array should have shape (number_of_samples, 4096)

    # 2. Analyze asymmetry for these DNA importance scores.
    # Since dna_importance_scores_per_position contains only non-negative values (it's a sum of absolute values),
    # using metric='sum_raw', 'sum_positive', or 'sum_absolute' on *these already processed scores*
    # will yield the same result for 'left_values' and 'right_values'.
    # 'sum_raw' is clear and sufficient here.
    print("\nAnalyzing asymmetry for Summed Absolute DNA Base Attributions:")
    dna_asymmetry_scores, dna_p_value = analyze_attribution_asymmetry(
        dna_importance_scores_per_position,
        data_label="Summed Absolute DNA Base Attribution",
        metric='sum_raw' 
    )

    # 3. Plot average half profiles for DNA importance
    plot_average_half_profiles(
        dna_importance_scores_per_position,
        data_label="Summed Absolute DNA Base Attribution"
    )

else:
    print("Warning: 'attrs_list' not found or not in the expected format (num_samples, 4, 4096). Skipping DNA base attribution asymmetry analysis.")

# For reference, how you might have called it for ATAC track attributions:
    print("\nAnalyzing asymmetry for ATAC Track Attributions (Sum of Raw):")
    analyze_attribution_asymmetry(
        atac_attribution_list,
        data_label="ATAC Track Attribution",
        metric='sum_raw'
    )




# %%
def plot_seqlet_with_atac(seqlets, attrs_list, atac_attribution_list, atac_pileup_list, 
                         sample_rank=0, context_size=20, colormap='RdBu_r', 
                         equal_color_scale=False, num_bins=None):
    """
    Create a two-panel plot with a NON-SYMMETRIC color-normalized heatmap.
    - Top: DNA base attributions (logo plot)  
    - Bottom: ATAC pileup with attribution heatmap background (0 is always white)
    
    The color scale for the heatmap now stretches to the true min and max of the data in the window.
    
    Args:
        ... (rest of the arguments) ...
        num_bins (int, optional): If specified, draws vertical dashed lines 
                                  to divide the plot into this many bins. 
                                  Defaults to None.
    """
    # --- This part of the function is unchanged ---
    sample = seqlets.iloc[[sample_rank]]
    slice_idx = int(sample['example_idx'].tolist()[0])
    sequence = sample['sequence'].tolist()[0]
    start = int(sample['start'].tolist()[0])
    end = int(sample['end'].tolist()[0])

    seqlet_center = (start + end) // 2
    seqlet_length = end - start
    total_window_size = seqlet_length + (2 * context_size)
    window_start = seqlet_center - (total_window_size // 2)
    window_end = seqlet_center + (total_window_size // 2)
    window_start = max(0, window_start)
    window_end = min(4096, window_end)
    if window_end - window_start < total_window_size:
        if window_start == 0:
            window_end = min(4096, window_start + total_window_size)
        elif window_end == 4096:
            window_start = max(0, window_end - total_window_size)
    
    print(f"Seqlet: {start}-{end} (center: {seqlet_center})")
    print(f"Window: {window_start}-{window_end} (size: {window_end - window_start})")
    if 'p-value' in sample.columns:
        p_value = sample['p-value'].tolist()[0]
        print(f"P-Value: {p_value}")
    
    plot_coords = np.arange(window_start, window_end)
    X_attr = attrs_list[slice_idx].astype(np.float64)
    atac_attr = atac_attribution_list[slice_idx].astype(np.float64)
    atac_pileup = atac_pileup_list[slice_idx].astype(np.float64)
    X_attr_windowed = X_attr[:, window_start:window_end]
    atac_attr_windowed = atac_attr[window_start:window_end]
    atac_pileup_windowed = atac_pileup[window_start:window_end]
    
    print(f"Windowed shapes: DNA={X_attr_windowed.shape}, ATAC_attr={atac_attr_windowed.shape}, ATAC_pileup={atac_pileup_windowed.shape}")

    fig = plt.figure(figsize=(18, 10), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[20, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.02)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    # --- Top panel and heatmap logic remains the same ---
    plot_logo(X_attr_windowed, ax=ax1)
    n_ticks = 8
    tick_positions = np.linspace(0, len(plot_coords)-1, n_ticks)
    tick_labels = np.linspace(plot_coords[0], plot_coords[-1], n_ticks).astype(int)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax1.set_xlabel("Genomic Coordinate")
    ax1.set_ylabel("DNA Attributions")
    ax1.set_title(f"DNA Base Attributions | Sample: {slice_idx} | {sequence}")

    heatmap_height = 25
    attr_heatmap = np.tile(atac_attr_windowed, (heatmap_height, 1))
    max_pileup = np.max(atac_pileup_windowed) if len(atac_pileup_windowed) > 0 else 1
    y_max = max_pileup * 1.1

    if atac_attr_windowed.size > 0:
        vmin_val = np.min(atac_attr_windowed)
        vmax_val = np.max(atac_attr_windowed)
    else:
        vmin_val, vmax_val = -1, 1

    if equal_color_scale:
        if atac_attr_windowed.size > 0:
            vabs_max = np.max(np.abs(atac_attr_windowed))
        else:
            vabs_max = 1
        vmin_val = -vabs_max
        vmax_val = vabs_max 
        
    norm = TwoSlopeNorm(vcenter=0, vmin=vmin_val, vmax=vmax_val)

    im = ax2.imshow(attr_heatmap, 
                    cmap=colormap,
                    aspect='auto',
                    extent=[plot_coords[0], plot_coords[-1], 0, y_max],
                    alpha=0.7,
                    interpolation='bilinear',
                    norm=norm)
    
    # --- The rest of the plotting logic is also unchanged ---
    ax2.plot(plot_coords, atac_pileup_windowed, color='black', linewidth=2.5, 
             label='ATAC-seq Pileup', alpha=0.9)
    ax2.set_xlim(plot_coords[0], plot_coords[-1])
    
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('ATAC Attribution', rotation=270, labelpad=15, fontsize=11)
    
    ax2.set_xlabel("Genomic Coordinate")
    ax2.set_ylabel("ATAC-seq Signal")
    ax2.set_title(f"ATAC Pileup with Attribution Heatmap | Sample: {slice_idx}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # <<< MODIFIED SECTION >>>
    # IF num_bins IS SPECIFIED, DRAW VERTICAL DASHED LINES
    if num_bins is not None and num_bins > 1:
        # Calculate the genomic coordinates for the bin dividers
        bin_edges = np.linspace(window_start, window_end, num_bins + 1)
        
        # Draw a line for each internal bin edge on both plots
        for edge in bin_edges[1:-1]:
            ax1.axvline(x=edge, color='grey', linestyle='--', linewidth=1.2, alpha=0.7)
            ax2.axvline(x=edge, color='grey', linestyle='--', linewidth=1.2, alpha=0.7)
    # <<< END OF MODIFIED SECTION >>>
    
    plt.show()

# %%
# === PWM-based scoring of top seqlets =====================================
# Author: automated addition

from typing import List, Tuple, Dict
import random


def load_jaspar_pwm(jaspar_path: str, pseudocount: float = 1e-2) -> np.ndarray:
    """Load a JASPAR motif and return a (4, L) probability matrix in A,C,G,T order."""
    base_order = ["A", "C", "G", "T"]
    rows: Dict[str, List[float]] = {b: [] for b in base_order}
    with open(jaspar_path) as f:
        for line in f:
            if line.startswith(">") or not line.strip():
                # skip header / empty lines
                continue
            base, rest = line.split("[", 1)
            counts = [float(x) for x in rest.strip("[] \n").split()]
            base_key = base.strip()
            if base_key in rows:
                rows[base_key].extend(counts)
    counts_arr = np.array([rows[b] for b in base_order], dtype=float) + pseudocount
    pwm = counts_arr / counts_arr.sum(axis=0, keepdims=True)  # column-wise normalisation
    return pwm


_BASE2IDX = {b: i for i, b in enumerate("ACGT")}
_COMP = str.maketrans("ACGT", "TGCA")


def _seq_logprob(seq: str, pwm: np.ndarray, offset: int) -> float:
    """Correctly calculate log-probability for a sequence at a given PWM offset."""
    idx = np.fromiter((_BASE2IDX.get(b, 0) for b in seq), dtype=int)
    # The columns in the PWM to select for each base in the sequence
    cols = np.arange(offset, offset + len(seq))
    # Select the probability of each base at its corresponding position and sum the logs.
    # The previous version incorrectly indexed a sub-matrix, this now correctly
    # selects the diagonal elements representing the actual match.
    return np.log(pwm[idx, cols] + 1e-9).sum()


def pwm_best_score(seq: str, pwm: np.ndarray) -> float:
    """Return best log-probability of *seq* (both strands) sliding along *pwm*."""
    L_pwm = pwm.shape[1]
    L_seq = len(seq)
    if L_seq > L_pwm:
        # If seqlet longer than PWM, truncate center to fit.
        start = (L_seq - L_pwm) // 2
        seq = seq[start : start + L_pwm]
        L_seq = len(seq)
    best = -np.inf
    # Forward strand
    for off in range(L_pwm - L_seq + 1):
        best = max(best, _seq_logprob(seq, pwm, off))
    # Reverse complement
    seq_rc = seq.translate(_COMP)[::-1]
    for off in range(L_pwm - L_seq + 1):
        best = max(best, _seq_logprob(seq_rc, pwm, off))
    return best


def score_and_display_seqlets(
    winning_seqlets,
    jaspar_path: str,
    motif_name: str,
    num_bins: int,
    score_top_n: int = None,
    display_top_n: int = 20,
    sort_by: str = "log-prob",
    num_permutations: int = 0,
    null_method: str = 'permute',
):
    """
    Loads a PWM, scores seqlets, and prints top results. Can also generate null
    distribution scores using one of two methods.

    Args:
        winning_seqlets: DataFrame containing seqlets and their 'bin_index'.
        jaspar_path: Path to the JASPAR motif file.
        motif_name: Name of the motif for labeling the output (e.g., "AR").
        num_bins: The total number of bins seqlets were sorted into.
        score_top_n (int, optional): The number of top frequent seqlets to score.
                                     If None, all unique seqlets are scored. Defaults to None.
        display_top_n (int): The number of top-scored seqlets to print for each bin.
                             Defaults to 20.
        sort_by (str): How to rank the final displayed list: 'log-prob' or 'frequency'.
        num_permutations (int): Number of random sequences to generate and score
                                for each occurrence of a seqlet. Defaults to 0.
        null_method (str): Method for generating the null distribution.
                             'permute': Shuffle the original seqlet (preserves
                             base composition).
                             'random': Generate a new random sequence of the same
                             length. Defaults to 'permute'.
    
    Returns:
        A tuple containing two lists:
        - Real scores: log-probs for actual seqlets, weighted by frequency.
        - Null scores: log-probs for the generated null sequences.
    """
    if winning_seqlets.empty:
        print(f"Skipping scoring for {motif_name}: `winning_seqlets` is empty.")
        return [], []

    score_label = f"top {score_top_n}" if score_top_n is not None else "all"
    print(f"\nScoring {score_label} most frequent seqlets per bin against {motif_name} PWM (Null method: '{null_method}')... 🔍")
    pwm = load_jaspar_pwm(jaspar_path)

    scores_to_return = []
    null_scores_to_return = []
    for b in range(num_bins):
        seqs_in_bin = winning_seqlets[winning_seqlets["bin_index"] == b]["sequence"]
        if seqs_in_bin.empty:
            continue
        
        # Get counts for all unique sequences in the bin
        seq_counts = seqs_in_bin.value_counts()
        
        # Decide which sequences to score based on score_top_n
        if score_top_n is not None:
            seqs_to_score = seq_counts.head(score_top_n)
        else:
            seqs_to_score = seq_counts
        
        scored: List[Tuple[str, int, float]] = []
        for seq, count in seqs_to_score.items():
            score = pwm_best_score(seq, pwm)
            scored.append((seq, count, score))
            # Append score `count` times for an accurate distribution plot
            scores_to_return.extend([score] * count)

            # Generate and score null distribution
            if num_permutations > 0:
                for _ in range(num_permutations):
                    # For each time the original seqlet appeared, generate one null version
                    for _ in range(count):
                        if null_method == 'permute':
                            null_seq_list = random.sample(seq, len(seq))
                            null_seq = "".join(null_seq_list)
                        elif null_method == 'random':
                            null_seq = "".join(random.choices("ACGT", k=len(seq)))
                        else:
                            raise ValueError(f"Unknown null_method: '{null_method}'. Choose 'permute' or 'random'.")
                        
                        null_score = pwm_best_score(null_seq, pwm)
                        null_scores_to_return.append(null_score)

        # --- Sorting Logic for Display ---
        if sort_by == "log-prob":
            # Sort by score (log-prob), descending
            scored.sort(key=lambda x: x[2], reverse=True)
            sort_label = "best-matching first"
        elif sort_by == "frequency":
            # The list is already sorted by frequency from value_counts()
            sort_label = "most frequent first"
        else:
            raise ValueError(f"Invalid sort_by value: '{sort_by}'. Must be 'log-prob' or 'frequency'.")

        print(f"\nBin {b}: Scored {len(scored)} unique seqlets vs {motif_name} → displaying top {min(display_top_n, len(scored))} ({sort_label})")
        for rank, (seq, count, sc) in enumerate(scored[:display_top_n], 1):
            print(f"  {rank:2d}. {seq:>12s} (n={count:<2})   log-prob = {sc:6.2f}")
    
    return scores_to_return, null_scores_to_return

# %%

# --- Step 0: Generate all possible seqlets from the entire dataset ---
print("Generating all seqlets from the dataset... 🧬")
all_seqlets = get_seqlets(attrs_list, method='recursive')
print(f"Found a total of {len(all_seqlets)} seqlets across all samples.")
print("-" * 20)


# --- Configuration ---
# Define which method to use for filtering samples:
# 'rank': Use the original method (bin must be in Top N).
# 'absolute': Bin's max signal must exceed a fixed value.
# 'relative': Bin's max signal must be >= X% of the sample's max signal.
METHOD = 'rank' # OPTIONS: 'rank', 'absolute', 'relative'

NUM_BINS = 5  # switched from 5 to 32 (~128 bp per bin across 4096 bp window)
TARGET_BIN_INDEX = 2

# --- Settings for 'rank' method ---
TOP_N_RANKING_ATTRIBUTION = 1   # Set to an integer (e.g., 1 for Top 1) or None to disable
TOP_N_RANKING_PILEUP = None
TOP_N_RANKING_ATTRS = None

# --- Settings for 'absolute' method ---
# Use the distribution plots from the next cell to help choose these values.
ABS_THRESHOLD_ATTRIBUTION = 0.5  # Example value; set to a number or None
ABS_THRESHOLD_PILEUP = None
ABS_THRESHOLD_ATTRS = None

# --- Settings for 'relative' method ---
# Values should be between 0.0 and 1.0 (e.g., 0.7 for 70%)
REL_THRESHOLD_ATTRIBUTION = 0.7 # Set to a float or None
REL_THRESHOLD_PILEUP = None
REL_THRESHOLD_ATTRS = None


# --- Step 1: Robustly bin the data ---
original_length = atac_attribution_list.shape[1]
trimmed_length = original_length - (original_length % NUM_BINS)
bin_size = trimmed_length // NUM_BINS

print(f"Original sequence length: {original_length}")
print(f"Number of bins: {NUM_BINS}")
print(f"Trimming sequences to length {trimmed_length} to create {NUM_BINS} equal bins of size {bin_size}bp.")
print("-" * 20)

# Trim and reshape 2D data arrays
trimmed_attributions = atac_attribution_list[:, :trimmed_length]
binned_attributions = trimmed_attributions.reshape(-1, NUM_BINS, bin_size)

trimmed_pileups = atac_pileup_list[:, :trimmed_length]
binned_pileups = trimmed_pileups.reshape(-1, NUM_BINS, bin_size)

# <<< FIX: Process 3D attrs_list >>>
# First, convert 3D attrs_list to 2D by taking the max across the base channels (axis=1)
attrs_list_2d = attrs_list.max(axis=1)

# Now, trim and reshape the new 2D array
trimmed_attrs = attrs_list_2d[:, :trimmed_length]
binned_attrs = trimmed_attrs.reshape(-1, NUM_BINS, bin_size)

# Calculate max values per bin
max_attributions_per_bin = binned_attributions.max(axis=2)
max_pileups_per_bin = binned_pileups.max(axis=2)
max_attrs_per_bin = binned_attrs.max(axis=2)


# --- Step 2 & 3: Find winning samples based on the selected METHOD ---
print(f"\n--- Analyzing Bin {TARGET_BIN_INDEX} using '{METHOD}' method ---")

final_mask = np.ones(atac_attribution_list.shape[0], dtype=bool)
active_filters = []

if METHOD == 'rank':
    # Determine the rank of each bin within each sample
    attribution_ranks = np.argsort(np.argsort(-max_attributions_per_bin, axis=1), axis=1)
    pileup_ranks = np.argsort(np.argsort(-max_pileups_per_bin, axis=1), axis=1)
    attrs_ranks = np.argsort(np.argsort(-max_attrs_per_bin, axis=1), axis=1)

    if TOP_N_RANKING_ATTRIBUTION is not None:
        mask = attribution_ranks[:, TARGET_BIN_INDEX] < TOP_N_RANKING_ATTRIBUTION
        final_mask &= mask
        active_filters.append(f"Attribution Rank < {TOP_N_RANKING_ATTRIBUTION}")
    if TOP_N_RANKING_PILEUP is not None:
        mask = pileup_ranks[:, TARGET_BIN_INDEX] < TOP_N_RANKING_PILEUP
        final_mask &= mask
        active_filters.append(f"Pileup Rank < {TOP_N_RANKING_PILEUP}")
    if TOP_N_RANKING_ATTRS is not None:
        mask = attrs_ranks[:, TARGET_BIN_INDEX] < TOP_N_RANKING_ATTRS
        final_mask &= mask
        active_filters.append(f"DNA Attrs Rank < {TOP_N_RANKING_ATTRS}")

elif METHOD == 'absolute':
    if ABS_THRESHOLD_ATTRIBUTION is not None:
        target_bin_vals = max_attributions_per_bin[:, TARGET_BIN_INDEX]
        final_mask &= (target_bin_vals >= ABS_THRESHOLD_ATTRIBUTION)
        active_filters.append(f"Attribution > {ABS_THRESHOLD_ATTRIBUTION}")
    if ABS_THRESHOLD_PILEUP is not None:
        target_bin_vals = max_pileups_per_bin[:, TARGET_BIN_INDEX]
        final_mask &= (target_bin_vals >= ABS_THRESHOLD_PILEUP)
        active_filters.append(f"Pileup > {ABS_THRESHOLD_PILEUP}")
    if ABS_THRESHOLD_ATTRS is not None:
        target_bin_vals = max_attrs_per_bin[:, TARGET_BIN_INDEX]
        final_mask &= (target_bin_vals >= ABS_THRESHOLD_ATTRS)
        active_filters.append(f"DNA Attrs > {ABS_THRESHOLD_ATTRS}")

elif METHOD == 'relative':
    if REL_THRESHOLD_ATTRIBUTION is not None:
        max_per_sample = max_attributions_per_bin.max(axis=1)
        thresholds = max_per_sample * REL_THRESHOLD_ATTRIBUTION
        target_bin_vals = max_attributions_per_bin[:, TARGET_BIN_INDEX]
        final_mask &= (target_bin_vals >= thresholds)
        active_filters.append(f"Attribution >= {REL_THRESHOLD_ATTRIBUTION*100}% of max")
    if REL_THRESHOLD_PILEUP is not None:
        max_per_sample = max_pileups_per_bin.max(axis=1)
        thresholds = max_per_sample * REL_THRESHOLD_PILEUP
        target_bin_vals = max_pileups_per_bin[:, TARGET_BIN_INDEX]
        final_mask &= (target_bin_vals >= thresholds)
        active_filters.append(f"Pileup >= {REL_THRESHOLD_PILEUP*100}% of max")
    if REL_THRESHOLD_ATTRS is not None:
        max_per_sample = max_attrs_per_bin.max(axis=1)
        thresholds = max_per_sample * REL_THRESHOLD_ATTRS
        target_bin_vals = max_attrs_per_bin[:, TARGET_BIN_INDEX]
        final_mask &= (target_bin_vals >= thresholds)
        active_filters.append(f"DNA Attrs >= {REL_THRESHOLD_ATTRS*100}% of max")

else:
    raise ValueError(f"Unknown METHOD: '{METHOD}'. Choose from 'rank', 'absolute', 'relative'.")


final_indices = np.where(final_mask)[0]

report_message = " AND ".join(active_filters) if active_filters else "No filters applied"

print(f"Found {len(final_indices)} samples where Bin {TARGET_BIN_INDEX} met the criteria for: {report_message}.")
print("These are the sample indices:")
print(final_indices)
print("-" * 20)


# --- Step 4: Find the Top 20 most common seqlets in each bin for the winning samples ---
# (This section remains unchanged as it correctly uses `final_indices`)
print("\n--- Finding Most Common Seqlets in Winning Samples --- 🏆")

# Filter the master seqlet list to include only those from our winning samples
winning_seqlets = all_seqlets[all_seqlets['example_idx'].isin(final_indices)].copy()

if winning_seqlets.empty:
    print("No seqlets were found in any of the winning samples that met the criteria.")
else:
    # Filter for seqlets >= 5bp long
    original_count = len(winning_seqlets)
    winning_seqlets = winning_seqlets[winning_seqlets['sequence'].str.len() >= 5]
    print(f"Filtered for seqlets >= 5bp long. Kept {len(winning_seqlets)} out of {original_count} winning seqlets.")

    # Determine which bin each seqlet belongs to based on its midpoint
    seqlet_midpoints = (winning_seqlets['start'] + winning_seqlets['end']) // 2
    winning_seqlets['bin_index'] = (seqlet_midpoints // bin_size).clip(upper=NUM_BINS - 1)

    # Find the most common seqlets in each bin
    for i in range(NUM_BINS):
        seqlets_in_bin = winning_seqlets[winning_seqlets['bin_index'] == i]

        if seqlets_in_bin.empty:
            print(f"\nBin {i}: No seqlets found.")
        else:
            sequence_counts = seqlets_in_bin['sequence'].value_counts()
            total_in_bin = len(seqlets_in_bin)
            print(f"\nBin {i}: Top seqlets (out of {total_in_bin} total):")
            top_seqlets = sequence_counts.head(20)
            for rank, (sequence, count) in enumerate(top_seqlets.items(), 1):
                print(f"  {rank:>2}. '{sequence}' (found {count} times)")


# %%
# === Visualize Distributions for Thresholding ===============================
# Author: automated addition

import seaborn as sns

def plot_max_value_distributions(
    max_attributions_per_bin,
    max_pileups_per_bin,
    max_attrs_per_bin,
    target_bin_index,
    num_bins
):
    """
    Plots the distributions of max values within bins to help choose thresholds.
    It shows the distribution for the target bin vs. all other bins.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), dpi=150)
    
    all_bins_label = f"Other Bins (not Bin {target_bin_index})"
    target_bin_label = f"Target Bin ({target_bin_index})"

    # --- Attribution Plot ---
    target_bin_attr = max_attributions_per_bin[:, target_bin_index]
    other_bins_attr = np.delete(max_attributions_per_bin, target_bin_index, axis=1).flatten()
    
    sns.kdeplot(other_bins_attr, ax=axes[0], label=all_bins_label, fill=True)
    sns.kdeplot(target_bin_attr, ax=axes[0], label=target_bin_label, fill=True, alpha=0.7)
    axes[0].set_title("Distribution of Max ATAC Attributions per Bin")
    axes[0].set_xlabel("Max ATAC Attribution Score in Bin")
    axes[0].legend()

    # --- Pileup Plot ---
    target_bin_pileup = max_pileups_per_bin[:, target_bin_index]
    other_bins_pileup = np.delete(max_pileups_per_bin, target_bin_index, axis=1).flatten()

    sns.kdeplot(other_bins_pileup, ax=axes[1], label=all_bins_label, fill=True)
    sns.kdeplot(target_bin_pileup, ax=axes[1], label=target_bin_label, fill=True, alpha=0.7)
    axes[1].set_title("Distribution of Max ATAC Pileups per Bin")
    axes[1].set_xlabel("Max ATAC Pileup in Bin")
    axes[1].legend()

    # --- DNA Attrs Plot ---
    target_bin_dna = max_attrs_per_bin[:, target_bin_index]
    other_bins_dna = np.delete(max_attrs_per_bin, target_bin_index, axis=1).flatten()
    
    sns.kdeplot(other_bins_dna, ax=axes[2], label=all_bins_label, fill=True)
    sns.kdeplot(target_bin_dna, ax=axes[2], label=target_bin_label, fill=True, alpha=0.7)
    axes[2].set_title("Distribution of Max DNA Attributions per Bin")
    axes[2].set_xlabel("Max DNA Attribution Score in Bin")
    axes[2].legend()
    
    plt.suptitle(f"Max Value Distributions: Comparing Target Bin {target_bin_index} to All Others", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

# Call the function using the data from the previous cell
if 'max_attributions_per_bin' in globals():
    plot_max_value_distributions(
        max_attributions_per_bin,
        max_pileups_per_bin,
        max_attrs_per_bin,
        TARGET_BIN_INDEX,
        NUM_BINS
    )
else:
    print("Run the previous cell to generate binned data before plotting.")


# %%
# === Main Analysis Function ===============================================

from scipy import stats

def analyze_motif_in_bin(
    winning_seqlets,
    motif_name: str,
    jaspar_path: str,
    target_bin_index: int,
    num_null_samples_per_seqlet: int = 1,
    null_method: str = 'random',
    display_top_n: int = 20,
):
    """
    Analyzes the top N most frequent seqlets from a specific bin against a
    given motif, performs a statistical test, and plots the resulting score
    distributions.
    """
    # 1. Filter for the target bin and find top N frequent seqlets
    seqlets_in_bin = winning_seqlets[winning_seqlets['bin_index'] == target_bin_index]
    
    if seqlets_in_bin.empty:
        print(f"No winning seqlets found in Bin {target_bin_index}. Skipping analysis for {motif_name}.")
        return

    sequence_counts = seqlets_in_bin['sequence'].value_counts()
    top_n_for_dist = sequence_counts.head(display_top_n)

    print(f"\n--- Analyzing Top {len(top_n_for_dist)} Frequent Seqlets for Motif '{motif_name}' in Bin {target_bin_index} ---")
    print(f"Found {len(seqlets_in_bin)} total seqlet occurrences in this bin.")

    # 2. Score all unique seqlets once for reporting best matches later
    pwm = load_jaspar_pwm(jaspar_path)
    scored_unique_seqlets: List[Tuple[str, int, float]] = []
    for seq, count in sequence_counts.items():
        score = pwm_best_score(seq, pwm)
        scored_unique_seqlets.append((seq, count, score))

    # 3. Build distributions using ONLY the top N frequent seqlets
    real_scores = []
    null_scores = []
    for seq, count in top_n_for_dist.items():
        # Find the pre-computed score for the sequence
        score = next((s for q, c, s in scored_unique_seqlets if q == seq), 0)
        real_scores.extend([score] * count)
        
        # Generate corresponding null scores
        for _ in range(num_null_samples_per_seqlet * count):
            if null_method == 'random':
                null_seq = "".join(random.choices("ACGT", k=len(seq)))
            elif null_method == 'permute':
                null_seq_list = random.sample(seq, len(seq))
                null_seq = "".join(null_seq_list)
            else:
                raise ValueError(f"Invalid null_method: '{null_method}'. Choose 'permute' or 'random'.")
            
            null_scores.append(pwm_best_score(null_seq, pwm))

    # 4. Statistical Test
    u_stat, p_value = None, None
    if len(real_scores) > 0 and len(null_scores) > 0:
        # Mann-Whitney U test: are scores from 'real_scores' stochastically greater than 'null_scores'?
        u_stat, p_value = stats.mannwhitneyu(real_scores, null_scores, alternative='greater')

    # 5. Visualization
    plt.figure(figsize=(12, 7))
    sns.kdeplot(null_scores, label=f"'{null_method.capitalize()}' Scores (Null)", fill=True, color="grey", alpha=0.5)
    sns.kdeplot(real_scores, label=f"Actual Scores for '{motif_name}'", fill=True, color="darkorange", alpha=0.6)
    
    mean_real = np.mean(real_scores)
    mean_null = np.mean(null_scores)
    
    plt.axvline(mean_null, color='black', linestyle='--', linewidth=2, label=f'Mean Null: {mean_null:.2f}')
    plt.axvline(mean_real, color='darkred', linestyle='--', linewidth=2, label=f'Mean Actual: {mean_real:.2f}')
    
    title = f"Significance of '{motif_name}' in Top {len(top_n_for_dist)} Frequent Seqlets from Bin {target_bin_index}\n"
    if p_value is not None:
        title += f"Mann-Whitney U p-value (Actual > Null): {p_value:.2e}"
    plt.title(title)
    plt.xlabel("Log-Probability Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 6. Report
    # Top N by frequency (already sorted in sequence_counts)
    top_by_frequency = sequence_counts.head(display_top_n)
    print(f"\n--- Top {display_top_n} Most FREQUENT Seqlets in Bin {target_bin_index} ---")
    for rank, (seq, count) in enumerate(top_by_frequency.items(), 1):
        print(f"  {rank:2d}. '{seq}' (found {count} times)")

    # Top N by PWM score
    scored_unique_seqlets.sort(key=lambda x: x[2], reverse=True)
    top_by_score = scored_unique_seqlets[:display_top_n]
    print(f"\n--- Top {display_top_n} Best-MATCHING Seqlets to '{motif_name}' in Bin {target_bin_index} ---")
    for rank, (seq, count, score) in enumerate(top_by_score, 1):
        print(f"  {rank:2d}. '{seq}' (n={count})   log-prob = {score:.2f}")

    print(f"\n--- Statistical Summary for {motif_name} in Bin {target_bin_index} ---")
    print(f" - Mean score of actual seqlets: {mean_real:.2f}")
    print(f" - Mean score of null sequences: {mean_null:.2f}")
    if p_value is not None:
        print(f" - Mann-Whitney U p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("   -> Result: The motif is SIGNIFICANTLY enriched in this bin.")
        else:
            print("   -> Result: The motif is NOT significantly enriched in this bin.")
    print("-" * 50)

# %%
# === Main Analysis: Test Motif Significance in a Specific Bin ==============

# --- Configuration ---
# 1. Define the motifs you want to be able to test.
MOTIF_DATABASE = {
    "AR": "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/analysis/interpretability/motifs/AR.jaspar", # From the first cell in the notebook
    "FOXA1": "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/analysis/interpretability/motifs/FOXA1.jaspar",
    "CTCF": "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/analysis/interpretability/motifs/CTCF.jaspar",
    "ERG": "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/analysis/interpretability/motifs/ERG.jaspar",
    "HOXB13": "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/analysis/interpretability/motifs/HOXB13.jaspar",
    "GATA1": "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/analysis/interpretability/motifs/GATA1.jaspar"
}

# 2. CHOOSE which motif and bin to analyze.
MOTIF_TO_ANALYZE = "FOXA1"
BIN_TO_ANALYZE = 0 # An integer from 0 to NUM_BINS-1

# --- Execution ---
if "winning_seqlets" in globals() and not winning_seqlets.empty:
    if MOTIF_TO_ANALYZE in MOTIF_DATABASE:
        analyze_motif_in_bin(
        winning_seqlets=winning_seqlets,
            motif_name=MOTIF_TO_ANALYZE,
            jaspar_path=MOTIF_DATABASE[MOTIF_TO_ANALYZE],
            target_bin_index=BIN_TO_ANALYZE,
            null_method='permute',
            display_top_n=20,
        )
    else:
        print(f"Error: Motif '{MOTIF_TO_ANALYZE}' not found in MOTIF_DATABASE.")
        print(f"Available motifs are: {list(MOTIF_DATABASE.keys())}")
else:
    print("Please run the preceding cells to generate the 'winning_seqlets' DataFrame first.")

# %%
# === Systematic Analysis: Spatial Significance Heatmap ======================
# Author: automated addition

import pandas as pd
from scipy import stats
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


import random

# Seed the random number generator
random.seed(42)

def _calculate_p_value_for_bin(
    winning_seqlets,
    jaspar_path: str,
    target_bin_index: int,
    num_bins: int,
    top_n_for_testing: int,
    null_method: str
) -> float:
    """
    A helper function that calculates the Mann-Whitney U p-value for a given
    motif in a specific bin without generating plots or text reports.
    Returns 1.0 if not significant or if no seqlets are found.
    """
    seqlets_in_bin = winning_seqlets[winning_seqlets['bin_index'] == target_bin_index]
    if seqlets_in_bin.empty:
        return 1.0

    sequence_counts = seqlets_in_bin['sequence'].value_counts()
    top_n_to_test = sequence_counts.head(top_n_for_testing)
    
    if top_n_to_test.empty:
        return 1.0

    # Score the top N seqlets and their null counterparts
    pwm = load_jaspar_pwm(jaspar_path)
    real_scores = []
    null_scores = []
    
    for seq, count in top_n_to_test.items():
        score = pwm_best_score(seq, pwm)
        real_scores.extend([score] * count)
        
        for _ in range(count): # Generate one null sequence per real one
            if null_method == 'random':
                null_seq = "".join(random.choices("ACGT", k=len(seq)))
            elif null_method == 'permute':
                null_seq = "".join(random.sample(seq, len(seq)))
            else:
                raise ValueError("Invalid null method")
            null_scores.append(pwm_best_score(null_seq, pwm))

    # Perform statistical test
    if not real_scores or not null_scores:
        return 1.0
    
    try:
        _, p_value = stats.mannwhitneyu(real_scores, null_scores, alternative='greater')
        return p_value
    except ValueError:
        # This can happen if all scores are identical
        return 1.0


def plot_spatial_significance_heatmap(
    winning_seqlets,
    motif_database,
    num_bins,
    top_n_for_testing=20,
    null_method='permute'
):
    """
    Generates a heatmap showing the statistical significance of multiple motifs
    across all spatial bins. Uses a binary color scheme and custom annotations
    (stars for significance, p-values otherwise).
    """
    print(f"--- Generating Spatial Significance Heatmap ---")
    print(f"Testing top {top_n_for_testing} seqlets per bin using '{null_method}' null method.")

    # Create a DataFrame to store p-values
    p_value_results = pd.DataFrame(
        index=motif_database.keys(),
        columns=range(num_bins),
        dtype=float
    )

    # Loop through each motif and bin to calculate significance
    for motif_name, jaspar_path in motif_database.items():
        print(f"Analyzing: {motif_name}...")
        for bin_idx in range(num_bins):
            p_val = _calculate_p_value_for_bin(
                winning_seqlets=winning_seqlets,
                jaspar_path=jaspar_path,
                target_bin_index=bin_idx,
                num_bins=num_bins,
                top_n_for_testing=top_n_for_testing,
                null_method=null_method
            )
            p_value_results.loc[motif_name, bin_idx] = p_val
    
    # --- Custom Visualization Logic ---

    # 1. Create a boolean matrix for significance (True if p < 0.05)
    significance_matrix = p_value_results < 0.05

    # 2. Create a matrix of custom string annotations
    def format_p_value_annotation(p):
        if p < 0.001: return '***'
        if p < 0.01: return '**'
        if p < 0.05: return '*'
        return f'{p:.2f}'
    
    annotation_matrix = p_value_results.applymap(format_p_value_annotation)

    # 3. Create a binary colormap
    # A simple light grey for non-significant, and a medium blue for significant
    cmap = ListedColormap(['#EAEAF2', '#5699C6'])

    # 4. Plot the heatmap
    plt.figure(figsize=(10, len(motif_database) * 1.5))
    ax = sns.heatmap(
        significance_matrix,    # Color cells based on True/False for significance
        annot=annotation_matrix,  # Use the custom strings for text
        fmt='s',                  # Tell heatmap the annotations are strings
        cmap=cmap,
        linewidths=1.5,
        linecolor='white',
        cbar=False,               # The binary color is self-explanatory
        annot_kws={"size": 12}    # Adjust font size for readability
    )
    
    # Add a legend manually for clarity
    not_sig_patch = mpatches.Patch(color='#EAEAF2', label='Not Significant (p ≥ 0.05)')
    sig_patch = mpatches.Patch(color='#5699C6', label='Significant (p < 0.05)')
    plt.legend(handles=[sig_patch, not_sig_patch], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    ax.set_title("Motif Significance Across Spatial Bins", fontsize=16, pad=20)
    ax.set_xlabel("Bin Index", fontsize=12)
    ax.set_ylabel("Motif", fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.show()

# --- Execution ---
if "winning_seqlets" in globals() and not winning_seqlets.empty:
    plot_spatial_significance_heatmap(
        winning_seqlets=winning_seqlets,
        motif_database=MOTIF_DATABASE,
        num_bins=NUM_BINS,
        top_n_for_testing=20,
        null_method='permute' # Using 'permute' is a stricter, more standard control here
    )
else:
    print("Please run the preceding cells to generate the 'winning_seqlets' DataFrame first.")

# %%



































# %%
# === Analysis of Negative Motifs (Repressors) ============================
# Author: automated addition

import os
import pandas as pd

print("\n" + "="*50)
print("=== Analyzing Negative Attributions for Repressive Motifs ===")
print("="*50 + "\n")

# --- Step 1: Generate seqlets from negative attribution scores ---
# By setting direction='negative', we look for "valleys" instead of "peaks".
print("Generating seqlets from negative attributions...")
negative_seqlets = get_seqlets(attrs_list, method='recursive', direction='negative')

if negative_seqlets.empty:
    print("No negative seqlets found.")
else:
    print(f"Found {len(negative_seqlets)} negative seqlets.")

    # --- Step 2: Find the most common negative seqlets ---
    # We can just look at the most frequent seqlets overall.
    negative_seqlet_counts = negative_seqlets['sequence'].value_counts()
    
    print("\n--- Top 20 Most Frequent Negative Seqlets ---")
    top_20_negative = negative_seqlet_counts.head(20)
    for rank, (sequence, count) in enumerate(top_20_negative.items(), 1):
        print(f"  {rank:2d}. '{sequence}' (found {count} times)")
    
    # --- Step 3: Optional - Score against a known repressor motif ---
    # Example: If you have a known repressor motif like REST, you could score against it.
    # This part is commented out but shows how you could extend the analysis.
    
    # To run this, you would need a JASPAR file for a known repressor motif.
    # REST_JASPAR_PATH = "/path/to/your/REST.jaspar" # <--- REPLACE WITH ACTUAL PATH
    
    # if 'REST_JASPAR_PATH' in locals() and os.path.exists(REST_JASPAR_PATH):
    #     print("\n--- Scoring negative seqlets against REST motif ---")
        
    #     # Add a dummy bin_index for the scoring function to work
    #     neg_seqlets_for_scoring = negative_seqlets.copy()
    #     neg_seqlets_for_scoring['bin_index'] = 0

    #     score_and_display_seqlets(
    #         neg_seqlets_for_scoring,
    #         jaspar_path=REST_JASPAR_PATH,
    #         motif_name="REST",
    #         num_bins=1, # We are not using bins here, so this is 1
    #         display_top_n=10,
    #         sort_by='log-prob'
    #     )
    # else:
    #     print("\nSkipping scoring against a repressor motif. To enable, provide a valid path in REST_JASPAR_PATH.")






# %%

# %%
# === Compare Negative Seqlets to Motif Database ===========================
# Author: automated addition

from scipy import stats

if 'negative_seqlets' in globals() and not negative_seqlets.empty:
    # Prepare DataFrame with a dummy bin index required by the scoring helper
    neg_seqlets_for_scoring = negative_seqlets.copy()
    neg_seqlets_for_scoring['bin_index'] = 0  # all in one bin

    print("\n" + "="*60)
    print("Comparing Negative Seqlets to Known Motifs (MOTIF_DATABASE)")
    print("="*60 + "\n")

    for motif_name, jaspar_path in MOTIF_DATABASE.items():
        print(f"\n--- {motif_name}: scoring negative seqlets ---")
        real_scores, null_scores = score_and_display_seqlets(
            winning_seqlets=neg_seqlets_for_scoring,
            jaspar_path=jaspar_path,
            motif_name=motif_name,
            num_bins=1,
            display_top_n=20,
            sort_by='log-prob',
            num_permutations=1,  # one permuted control per occurrence
            null_method='permute'
        )

        # Compute significance if we obtained both score lists
        if real_scores and null_scores:
            try:
                u_stat, p_value = stats.mannwhitneyu(real_scores, null_scores, alternative='greater')
                print(f"Mann-Whitney U p-value (Actual > Null): {p_value:.2e}")
                if p_value < 0.05:
                    print("  -> Significant enrichment among negative seqlets.")
                else:
                    print("  -> No significant enrichment detected.")
            except ValueError as e:
                # This can happen if all scores are identical
                print(f"Could not compute significance test: {e}")
        else:
            print("Insufficient score data to compute significance.")
else:
    print("No negative seqlets available. Run the previous cell to generate them first.")



# %%
# === Spatial Significance Heatmap for Negative Seqlets ====================
# Author: automated addition

if 'negative_seqlets' in globals() and not negative_seqlets.empty:
    print("\n" + "="*60)
    print("Generating Spatial Significance Heatmap for NEGATIVE Seqlets")
    print("="*60 + "\n")

    # Ensure bin_size and NUM_BINS are defined (from earlier data prep)  
    if 'bin_size' not in globals() or 'NUM_BINS' not in globals():
        raise RuntimeError("bin_size and NUM_BINS must be defined from the earlier data-binning step.")

    # Assign each negative seqlet to a spatial bin based on its midpoint
    neg_seqlets_binned = negative_seqlets.copy()
    seqlet_midpoints = (neg_seqlets_binned['start'] + neg_seqlets_binned['end']) // 2
    neg_seqlets_binned['bin_index'] = (seqlet_midpoints // bin_size).clip(upper=NUM_BINS - 1)

    # Use existing heatmap function
    plot_spatial_significance_heatmap(
        winning_seqlets=neg_seqlets_binned,
        motif_database=MOTIF_DATABASE,
        num_bins=NUM_BINS,
        top_n_for_testing=20,
        null_method='permute'
    )
else:
    print("No negative seqlets available. Run the negative seqlet generation cell first.")

# %%
# === Motif Proximity Analysis =============================================
# Author: automated addition

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import random

def analyze_motif_proximity(
    all_seqlets_pd: pd.DataFrame,
    motif_database: dict,
    motif1_name: str,
    motif2_name: str,
    logp_cutoff: float = -10.0,
    proximity_threshold: int = 25,
    num_permutations: int = 100,
):
    """
    Analyzes the spatial proximity of two motifs (e.g., AR and FOXA1).

    This function performs the following steps:
    1.  Labels all seqlets with their best-matching motif if the PWM score
        is above a specified cutoff.
    2.  For each 4096-bp sample window, calculates the distances (gaps)
        between all pairs of motif1 and motif2 seqlets.
    3.  Plots a histogram of the observed gap distances.
    4.  Performs a permutation test to determine if the number of "close"
        pairs (gap <= proximity_threshold) is statistically significant
        compared to a null model where motif positions are shuffled.

    Args:
        all_seqlets_pd: Pandas DataFrame of all seqlets, from get_seqlets().
        motif_database: Dictionary mapping motif names to their JASPAR file paths.
        motif1_name: The name of the first motif (e.g., "AR").
        motif2_name: The name of the second motif (e.g., "FOXA1").
        logp_cutoff: The log-probability score required for a seqlet to be
                     considered a match to a motif.
        proximity_threshold: The distance in base pairs to define a "close" pair.
        num_permutations: The number of shuffling iterations for the permutation test.
    """
    print("\n" + "="*60)
    print(f"Analyzing Proximity between '{motif1_name}' and '{motif2_name}' Motifs")
    print("="*60 + "\n")

    # --- 1. Annotate every seqlet with its best-matching motif ---
    print(f"Scoring seqlets against PWMs (log-prob cutoff: {logp_cutoff})...")
    pwm1 = load_jaspar_pwm(motif_database[motif1_name])
    pwm2 = load_jaspar_pwm(motif_database[motif2_name])

    def label_seqlet(seq: str) -> str:
        score1 = pwm_best_score(seq, pwm1)
        score2 = pwm_best_score(seq, pwm2)
        if max(score1, score2) < logp_cutoff:
            return None  # Not a good match to either
        return motif1_name if score1 > score2 else motif2_name

    all_seqlets_pd['motif'] = all_seqlets_pd['sequence'].apply(label_seqlet)
    motif_seqlets_pl = pl.from_pandas(all_seqlets_pd.dropna(subset=['motif']))
    
    count1 = motif_seqlets_pl.filter(pl.col('motif') == motif1_name).height
    count2 = motif_seqlets_pl.filter(pl.col('motif') == motif2_name).height
    print(f"Found {count1} '{motif1_name}' seqlets and {count2} '{motif2_name}' seqlets.")
    
    if count1 == 0 or count2 == 0:
        print("One or both motifs not found in seqlets. Cannot perform proximity analysis.")
        return

    # --- 2. Compute all pairwise distances between the two motifs ---
    def get_pairwise_distances(df: pl.DataFrame) -> list:
        """Helper to return a list of gaps for one 4096-bp region."""
        m1 = df.filter(pl.col("motif") == motif1_name).select(["start", "end"]).to_numpy()
        m2 = df.filter(pl.col("motif") == motif2_name).select(["start", "end"]).to_numpy()
        gaps = []
        if m1.size == 0 or m2.size == 0:
            return gaps
        for s1, e1 in m1:
            for s2, e2 in m2:
                gap = max(0, max(s1, s2) - min(e1, e2) - 1)  # 0=overlap
                gaps.append(gap)
        return gaps

    print("\nCalculating observed distances between motifs in each sample...")
    observed_gaps = []
    for _, group_df in motif_seqlets_pl.group_by("example_idx"):
        observed_gaps.extend(get_pairwise_distances(group_df))
    
    observed_gaps = np.array(observed_gaps)
    if observed_gaps.size == 0:
        print("No samples contained both motifs. Cannot perform proximity analysis.")
        return
        
    print(f"Collected {len(observed_gaps)} '{motif1_name}'<->'{motif2_name}' pairs across all samples.")

    # --- 3. Visualize the distribution of distances ---
    plt.figure(figsize=(10, 6))
    plt.hist(observed_gaps, bins=np.logspace(0, np.log10(4096), 50), color='steelblue', alpha=0.8)
    plt.xscale('log')
    plt.axvline(proximity_threshold, color='red', linestyle='--', label=f'Proximity Threshold ({proximity_threshold} bp)')
    plt.title(f"Distribution of Distances between '{motif1_name}' and '{motif2_name}'")
    plt.xlabel("Distance (bp, log scale)")
    plt.ylabel("Number of Motif Pairs")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # --- 4. Perform permutation test for enrichment of close pairs ---
    print(f"\nPerforming permutation test for enrichment of close pairs (<= {proximity_threshold} bp)...")
    
    def count_close_pairs_in_shuffled(group_df: pl.DataFrame, rng) -> int:
        """Shuffle motif1 positions and count close pairs."""
        m1 = group_df.filter(pl.col("motif") == motif1_name)
        m2 = group_df.filter(pl.col("motif") == motif2_name)
        
        if m1.height == 0 or m2.height == 0:
            return 0
        
        # Shuffle starts and recalculate ends
        shuffled_starts = rng.permutation(m1['start'].to_numpy())
        m1_shuffled = m1.with_columns(
            pl.Series("start", shuffled_starts),
            pl.Series("end", shuffled_starts + (m1['end'] - m1['start']))
        )
        
        # Re-run distance calculation
        shuffled_distances = get_pairwise_distances(pl.concat([m1_shuffled, m2]))
        return np.sum(np.array(shuffled_distances) <= proximity_threshold)

    null_distribution = []
    grouped_data = list(motif_seqlets_pl.group_by("example_idx"))
    
    for i in tqdm(range(num_permutations), desc="Permutation Test"):
        rng = np.random.default_rng(seed=i)
        total_close_in_perm = 0
        for _, group_df in grouped_data:
            total_close_in_perm += count_close_pairs_in_shuffled(group_df, rng)
        null_distribution.append(total_close_in_perm)

    observed_close_count = np.sum(observed_gaps <= proximity_threshold)
    p_value = (np.sum(np.array(null_distribution) >= observed_close_count) + 1) / (num_permutations + 1)

    print("\n--- Proximity Analysis Results ---")
    print(f"Observed pairs within {proximity_threshold} bp: {observed_close_count}")
    print(f"Mean close pairs in permutations: {np.mean(null_distribution):.2f}")
    print(f"Permutation p-value (enrichment): {p_value:.4f}")
    if p_value < 0.05:
        print("Result: The motifs are SIGNIFICANTLY closer than expected by chance.")
    else:
        print("Result: The proximity of motifs is NOT statistically significant.")
    print("-" * 60)

# %%
# === Run Motif Proximity Analysis =========================================
# Ensure 'all_seqlets' and 'MOTIF_DATABASE' are defined from previous cells.
from tqdm import tqdm

if 'all_seqlets' in globals() and 'MOTIF_DATABASE' in globals():
    analyze_motif_proximity(
        all_seqlets_pd=all_seqlets,
        motif_database=MOTIF_DATABASE,
        motif1_name="AR",
        motif2_name="FOXA1",
        logp_cutoff=-8.0, # A stricter cutoff might give cleaner results
        proximity_threshold=50, # How close is "close"? (in bp)
        num_permutations=100
    )
else:
    print("Please run the cells that define 'all_seqlets' and 'MOTIF_DATABASE' first.")

# %%
