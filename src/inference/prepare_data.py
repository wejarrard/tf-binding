import os
import gzip
import shutil
import argparse
import time
from pathlib import Path
from random import random, randrange
from tqdm import tqdm
import numpy as np
import polars as pl
import pysam
import torch
import torch.nn.functional as F
from einops import rearrange
from pyfaidx import Fasta
from torch.utils.data import DataLoader, Dataset
import orjson

# Set multiprocessing sharing strategy
torch.multiprocessing.set_sharing_strategy('file_system')

# Helper functions
def exists(val):
    return val is not None

def identity(t):
    return t

def cast_list(t):
    return t if isinstance(t, list) else [t]

def coin_flip():
    return random() > 0.5

# genomic function transforms
seq_indices_embed = torch.zeros(256).long()
seq_indices_embed[ord("a")] = 0
seq_indices_embed[ord("c")] = 1
seq_indices_embed[ord("g")] = 2
seq_indices_embed[ord("t")] = 3
seq_indices_embed[ord("n")] = 4
seq_indices_embed[ord("A")] = 0
seq_indices_embed[ord("C")] = 1
seq_indices_embed[ord("G")] = 2
seq_indices_embed[ord("T")] = 3
seq_indices_embed[ord("N")] = 4
seq_indices_embed[ord(".")] = -1

one_hot_embed = torch.zeros(256, 4)
one_hot_embed[ord("a")] = torch.Tensor([1.0, 0.0, 0.0, 0.0])
one_hot_embed[ord("c")] = torch.Tensor([0.0, 1.0, 0.0, 0.0])
one_hot_embed[ord("g")] = torch.Tensor([0.0, 0.0, 1.0, 0.0])
one_hot_embed[ord("t")] = torch.Tensor([0.0, 0.0, 0.0, 1.0])
one_hot_embed[ord("n")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
one_hot_embed[ord("A")] = torch.Tensor([1.0, 0.0, 0.0, 0.0])
one_hot_embed[ord("C")] = torch.Tensor([0.0, 1.0, 0.0, 0.0])
one_hot_embed[ord("G")] = torch.Tensor([0.0, 0.0, 1.0, 0.0])
one_hot_embed[ord("T")] = torch.Tensor([0.0, 0.0, 0.0, 1.0])
one_hot_embed[ord("N")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
one_hot_embed[ord(".")] = torch.Tensor([0.25, 0.25, 0.25, 0.25])

reverse_complement_map = torch.Tensor([3, 2, 1, 0, 4]).long()

def torch_fromstring(seq_strs):
    batched = not isinstance(seq_strs, str)
    seq_strs = cast_list(seq_strs)
    np_seq_chrs = list(map(lambda t: np.frombuffer(t.encode(), dtype=np.uint8), seq_strs))
    seq_chrs = list(map(lambda t: torch.from_numpy(t.copy()), np_seq_chrs))
    return torch.stack(seq_chrs) if batched else seq_chrs[0]

def str_to_seq_indices(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return seq_indices_embed[seq_chrs.long()]

def str_to_one_hot(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return one_hot_embed[seq_chrs.long()]

def seq_indices_to_one_hot(t, padding=-1):
    is_padding = t == padding
    t = t.clamp(min=0)
    one_hot = F.one_hot(t, num_classes=5)
    out = one_hot[..., :4].float().masked_fill(is_padding[..., None], 0.25)
    return out

# Augmentations
def seq_indices_reverse_complement(seq_indices):
    return torch.flip(reverse_complement_map[seq_indices.long()], dims=(-1,))

def one_hot_reverse_complement(one_hot):
    return torch.flip(one_hot, (-1, -2))

# Pileup processing
def process_pileups(pileup_dir: Path, chr_name: str, start: int, end: int):
    pileup_file = pileup_dir / f"{chr_name}.pileup.gz"
    assert pileup_file.exists(), f"Pileup file {pileup_file} does not exist"

    tabixfile = pysam.TabixFile(str(pileup_file))
    records = [rec.split("\t") for rec in tabixfile.fetch(chr_name, start, end)]
    return pl.DataFrame({
        "chr_name": [rec[0] for rec in records],
        "position": [int(rec[1]) for rec in records],
        "nucleotide": [rec[2] for rec in records],
        "count": [float(rec[3]) for rec in records],
    })

def mask_sequence(input_tensor, mask_prob=0.15, mask_value=-1):
    labels = input_tensor.clone()
    mask_rows = torch.bernoulli(torch.ones((input_tensor.shape[0], 1)) * mask_prob).bool()
    mask = mask_rows.expand_as(input_tensor)
    masked_tensor = input_tensor.clone()
    masked_tensor[mask] = mask_value
    labels[~mask] = -1
    return masked_tensor, labels


def gaussian_smooth_1d(values: torch.Tensor, kernel_size: int = 15, sigma: float = 2.0) -> torch.Tensor:
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel = torch.exp(-torch.arange(-(kernel_size // 2), kernel_size // 2 + 1) ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)
    values = values.view(1, 1, -1)
    padding = (kernel_size - 1) // 2
    smoothed = F.conv1d(values, kernel, padding=padding)
    return smoothed.view(-1, 1)

class GenomicInterval:
    def __init__(self, fasta_file, context_length=None, return_seq_indices=False, shift_augs=None, rc_aug=False):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), "Path to FASTA file must exist"
        self.fasta_path = str(fasta_file)
        self.return_seq_indices = return_seq_indices
        self.context_length = context_length
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug

    @property
    def seqs(self):
        return Fasta(self.fasta_path)

    def __call__(self, chr_name, start, end, pileup_dir, return_augs=False):
        interval_length = end - start
        seqs = self.seqs

        try:
            chromosome = seqs[chr_name]
        except KeyError:
            chromosome = seqs[chr_name[3:]]

        chromosome_length = len(chromosome)
        rand_shift = 0

        # Shift augmentations
        if exists(self.shift_augs):
            min_shift, max_shift = self.shift_augs
            max_shift += 1
            min_shift = max(start + min_shift, 0) - start
            max_shift = min(end + max_shift, chromosome_length) - end
            rand_shift = randrange(min_shift, max_shift)
            start += rand_shift
            end += rand_shift

        left_padding = right_padding = 0
        if exists(self.context_length) and interval_length < self.context_length:
            extra_seq = self.context_length - interval_length
            extra_left_seq = extra_seq // 2
            extra_right_seq = extra_seq - extra_left_seq
            start -= extra_left_seq
            end += extra_right_seq

        if start < 0:
            left_padding = -start
            start = 0

        if end > chromosome_length:
            right_padding = end - chromosome_length
            end = chromosome_length

        seq = ("." * left_padding) + str(chromosome[start:end]) + ("." * right_padding)
        should_rc_aug = self.rc_aug and coin_flip()

        if self.return_seq_indices:
            seq = str_to_seq_indices(seq)
            if should_rc_aug:
                seq = seq_indices_reverse_complement(seq)
            return seq

        one_hot = str_to_one_hot(seq)
        if should_rc_aug:
            one_hot = one_hot_reverse_complement(one_hot)

        # Replace the pileup processing section with:
        reads_tensor = torch.zeros((one_hot.shape[0], 1), dtype=torch.float)
        extended_data = torch.cat((one_hot, reads_tensor), dim=-1)
        df = process_pileups(pileup_dir, chr_name, start, end)

        max_count = df["count"].max() if df.height > 0 else 1
        min_count = df["count"].min() if df.height > 0 else 0
        min_range = -1
        max_range = 1
        range_size = max_range - min_range
        count_range = max(max_count - min_count, 1)

        for row in df.iter_rows(named=True):
            position = row["position"]
            count = row["count"]
            relative_position = position - start - 1
            scaled_count = (count - min_count) / count_range * range_size + min_range
            reads_tensor[relative_position, 0] = scaled_count

        # Apply Gaussian smoothing
        smoothed_reads = gaussian_smooth_1d(reads_tensor)
        extended_data = torch.cat((one_hot, smoothed_reads), dim=-1)

        if not return_augs:
            return extended_data

        rand_shift_tensor = torch.tensor([rand_shift])
        rand_aug_bool_tensor = torch.tensor([should_rc_aug])
        return extended_data, rand_shift_tensor, rand_aug_bool_tensor

class TFIntervalDataset(Dataset):
    def __init__(self, bed_file, fasta_file, cell_lines_dir, filter_df_fn=identity, chr_bed_to_fasta_map=dict(), mode="train", num_tfs=2, context_length=None, return_seq_indices=False, shift_augs=None, rc_aug=False, return_augs=False):
        super().__init__()
        bed_path = Path(bed_file)
        assert bed_path.exists(), "Path to .bed file must exist"

        df = pl.read_csv(str(bed_path), separator="\t")
        df = filter_df_fn(df)
        self.df = df
        self.num_tfs = num_tfs
        self.chr_bed_to_fasta_map = chr_bed_to_fasta_map
        self.return_augs = return_augs
        self.cell_lines_dir = Path(cell_lines_dir)

        self.processor = GenomicInterval(
            fasta_file=fasta_file,
            context_length=context_length,
            return_seq_indices=return_seq_indices,
            shift_augs=shift_augs,
            rc_aug=rc_aug,
        )
        self.label_folders = sorted([f.name for f in self.cell_lines_dir.iterdir() if f.is_dir()], key=lambda x: x)
        self.mode = mode

    def process_tfs(self, score: int, label: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([score]), torch.tensor([label], dtype=torch.float32)

    def __getitem__(self, ind):
        interval = self.df.row(ind)
        chr_name, start, end, score, label, cell_line, motifs, motif_score = interval
        chr_name = self.chr_bed_to_fasta_map.get(chr_name, chr_name)
        score, label_encoded = self.process_tfs(score, label)

        pileup_dir = self.cell_lines_dir / Path(cell_line) / "pileup_mod"
        data = self.processor(chr_name, start, end, pileup_dir, return_augs=self.return_augs)

        if self.mode == "train":
            return data, label_encoded, score
        elif self.mode == "inference":
            return data, label_encoded, score, chr_name, start, end, cell_line, motifs, motif_score
        else:
            return data

    def __len__(self):
        return len(self.df)

def main(compression: str, max_file_size: int, data_dir: str, output_path: str, input_file: str, cell_line_dir: str):
    base_filename = os.path.join(output_path, "dataset")
    file_index = 1
    filename = f"{base_filename}_{file_index}.jsonl"
    if compression == 'gzip':
        filename += '.gz'
    dataset = TFIntervalDataset(
        bed_file=os.path.join(data_dir, 'data_splits', input_file),
        fasta_file=os.path.join(data_dir, "genome.fa"),
        cell_lines_dir=os.path.join(cell_line_dir),
        return_augs=False,
        rc_aug=False,
        shift_augs=(0, 0),
        context_length=4_096,
        mode="inference",
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    max_file_size_bytes = max_file_size * 1024 * 1024  # Convert MB to bytes

    def open_file(filename, mode='wb'):
        return gzip.open(filename, mode) if compression == 'gzip' else open(filename, mode)

    jsonl_file = open_file(filename)

    try:
        for item in tqdm(dataloader, desc="Preparing Data"):
            data_point = {
                "input": item[0].numpy().tolist(),
                "target": item[1].numpy().tolist(),
                "weight": item[2].numpy().tolist(),
                "chr_name": item[3][0],
                "start": item[4].item(),
                "end": item[5].item(),
                "cell_line": item[6][0],
                "motifs": item[7][0],
                "motif_score": item[8].numpy().tolist()
            }
            json_bytes = orjson.dumps(data_point) + b'\n'
            jsonl_file.write(json_bytes)

            if os.path.getsize(filename) > max_file_size_bytes:
                jsonl_file.close()
                file_index += 1
                filename = f"{base_filename}_{file_index}.jsonl"
                if compression == 'gzip':
                    filename += '.gz'
                jsonl_file = open_file(filename)
    finally:
        jsonl_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for TF binding.")
    
    # Compression and File Size Configurations
    parser.add_argument('--compression', choices=['gzip', 'none'], default='gzip', help="Specify compression type (gzip or none) (default: 'gzip')")
    parser.add_argument('--max_file_size', type=int, default=5, help="Specify maximum file size in MB (default: 5)")
    
    # Directory and File Configurations
    parser.add_argument('--data_dir', type=str, default="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data", help='Specify the data directory (default: "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data")')
    parser.add_argument('--output_path', type=str, default="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/jsonl", help='Specify the output path (default: "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/jsonl")')
    parser.add_argument('--input_file', type=str, default="validation_combined.csv", help='Specify the input CSV file name (default: "validation_combined.csv")')
    parser.add_argument('--cell_line_dir', type=str, default="/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines", help='Specify the cell line directory (default: "/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines")')

    args = parser.parse_args()

    # Ensure the output directory exists and remove existing files
    os.makedirs(args.output_path, exist_ok=True)
    for file in os.listdir(args.output_path):
        file_path = os.path.join(args.output_path, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
                print(f"Removed file: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Removed directory and its contents: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    main(
        compression=args.compression,
        max_file_size=args.max_file_size,
        data_dir=args.data_dir,
        output_path=args.output_path,
        input_file=args.input_file,
        cell_line_dir=args.cell_line_dir
    )