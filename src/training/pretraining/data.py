import ast
import os
import time
from pathlib import Path
from random import random, randrange
from enum import Enum, auto

import numpy as np
import polars as pl
import pysam
import torch
import torch.nn.functional as F
from einops import rearrange
from pyfaidx import Fasta
from torch.utils.data import DataLoader, Dataset
from scipy.signal import savgol_coeffs


CELL_LINES = [
    "22Rv1",
    "LNCAP",
    "PC-3",
    "NCI-H660",
    "C42B",
    "C4-2",
    "MCF7",
    "Ramos",
    "A549",
    "HT-1376",
    "K-562",
    "JURKAT",
    "Hep_G2",
    "MCF_10A",
    "SCaBER",
    "SEM",
    "786-O",
    "Ishikawa",
    "MOLT-4",
    "BJ_hTERT",
    "SIHA",
    "Detroit_562",
    "OVCAR-8",
    "PANC-1",
    "NCI-H69",
    "HELA",
    "HuH-7",
    "K-562",
    "THP-1",
    "SK-N-SH",
    "U-87_MG",
    "RS411",
    "TC-32",
    "TTC1240",
    "VCAP",
]

class FilterType(Enum):
    NONE = auto()
    GAUSSIAN = auto()
    SAVGOL = auto()

    def __str__(self):
        return self.name.lower()

    @classmethod
    def from_str(cls, value: str):
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"Invalid FilterType: {value}")

class TransformType(Enum):
    NONE = auto()
    LOG10 = auto()
    MINMAX = auto()
    LOG10_MINMAX = auto()

    def __str__(self):
        return self.name.lower()

    @classmethod
    def from_str(cls, value: str):
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"Invalid TransformType: {value}")
    
    

class Mode(Enum):
    TRAIN = auto()
    VALIDATION = auto()
    INFERENCE = auto()

# helper functions


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
    np_seq_chrs = list(
        map(lambda t: np.frombuffer(t.encode(), dtype=np.uint8), seq_strs)
    )
    seq_chrs = list(
        map(lambda t: torch.from_numpy(t.copy()), np_seq_chrs)
    )  # Updated to copy the numpy array
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
    out = one_hot[..., :4].float()
    out = out.masked_fill(is_padding[..., None], 0.25)
    return out


# augmentations


def seq_indices_reverse_complement(seq_indices):
    complement = reverse_complement_map[seq_indices.long()]
    return torch.flip(complement, dims=(-1,))


def one_hot_reverse_complement(one_hot):
    *_, n, d = one_hot.shape
    assert d == 4, "must be one hot encoding with last dimension equal to 4"
    return torch.flip(one_hot, (-1, -2))


import torch.nn.functional as F

def gaussian_smooth_1d(values: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """
    Apply 1D Gaussian smoothing to a tensor of values.
    
    Args:
        values: Input tensor of shape (sequence_length, 1)
        kernel_size: Size of Gaussian kernel (should be odd)
        sigma: Standard deviation of Gaussian kernel
    
    Returns:
        Smoothed tensor of same shape as input
    """
    # Ensure kernel size is odd
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # Create Gaussian kernel with explicit dtype
    kernel = torch.exp(-torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    
    # Ensure both kernel and values are float32
    kernel = kernel.float()
    values = values.float()
    
    # Reshape kernel for 1D convolution
    kernel = kernel.view(1, 1, -1)
    
    # Reshape input for convolution
    values = values.view(1, 1, -1)
    
    # Apply padding to maintain sequence length
    padding = (kernel_size - 1) // 2
    
    # Perform convolution
    smoothed = F.conv1d(values, kernel, padding=padding)
    
    return smoothed.view(-1, 1)


def savgol_smooth_1d(values: torch.Tensor, window_length: int = 5, polyorder: int = 2) -> torch.Tensor:
    """
    Apply 1D Savitzky-Golay smoothing to a tensor of values.
    
    Args:
        values: Input tensor of shape (sequence_length, 1)
        window_length: Length of the filter window (should be odd)
        polyorder: Order of the polynomial used to fit the samples (must be less than window_length)
    
    Returns:
        Smoothed tensor of same shape as input
    """
    # Ensure window length is odd
    window_length = window_length if window_length % 2 == 1 else window_length + 1
    
    # Get Savitzky-Golay coefficients and convert to float32
    coeffs = torch.tensor(savgol_coeffs(window_length, polyorder), dtype=torch.float32, device=values.device)
    
    # Ensure values are float32
    values = values.float()
    
    # Reshape kernel for 1D convolution
    kernel = coeffs.view(1, 1, -1)
    
    # Reshape input for convolution
    values = values.view(1, 1, -1)
    
    # Apply padding to maintain sequence length
    padding = (window_length - 1) // 2
    
    # Perform convolution
    smoothed = F.conv1d(values, kernel, padding=padding)
    
    return smoothed.view(-1, 1)


# PILEUP PROCESSING
def process_pileups(pileup_dir: Path, chr_name: str, start: int, end: int):
    pileup_file = pileup_dir / f"{chr_name}.pileup.gz"

    assert pileup_file.exists(), f"pileup file for {pileup_file} does not exist"

    tabixfile = pysam.TabixFile(str(pileup_file))

    records = []
    for rec in tabixfile.fetch(chr_name, start, end):
        records.append(rec.split("\t"))

    # Convert records to a DataFrame using Polars:
    df = pl.DataFrame(
        {
            "chr_name": [rec[0] for rec in records],
            "position": [int(rec[1]) for rec in records],
            "nucleotide": [rec[2] for rec in records],
            "count": [float(rec[3]) for rec in records],
        }
    )

    return df


def mask_sequence(input_tensor, mask_prob=0.15, mask_value=-1):
    """
    Masks the input sequence tensor with given probability.
    Masks the entire row including all columns.
    """
    # Clone the input tensor to create labels
    labels = input_tensor.clone()

    # Calculate row mask
    mask_rows = torch.bernoulli(
        torch.ones((input_tensor.shape[0], 1)) * mask_prob
    ).bool()

    # Expand mask to all columns
    mask = mask_rows.expand_as(input_tensor)

    # Apply mask to input_tensor to create the masked tensor
    masked_tensor = input_tensor.clone()
    masked_tensor[mask] = mask_value

    # Set the labels where mask is not True to -1 (or any invalid label)
    labels[~mask] = -1  # only calculate loss on masked tokens

    return masked_tensor, labels

class GenomicInterval:
    def __init__(
        self,
        *,
        fasta_file,
        context_length=None,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
        filter_type=FilterType.NONE,
        filter_params=None,      
        transform_type=TransformType.NONE,
        transform_params=None,
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), "path to fasta file must exist"
        self.fasta_path = str(fasta_file)
        self.return_seq_indices = return_seq_indices
        self.context_length = context_length
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug
        
        # Validate filter_type is a FilterType enum
        if isinstance(filter_type, str):
            try:
                filter_type = FilterType[filter_type.upper()]
            except KeyError:
                raise ValueError(f"Invalid filter type: {filter_type}. Must be one of {[f.name for f in FilterType]}")
        elif not isinstance(filter_type, FilterType):
            raise ValueError(f"filter_type must be a FilterType enum or string, got {type(filter_type)}")
            
        self.filter_type = filter_type
        self.filter_params = filter_params or {}
        self.transform_type = transform_type
        self.transform_params = transform_params

    def apply_smoothing(self, reads_tensor):
        """Apply the selected smoothing filter to the reads tensor."""
        if self.filter_type == FilterType.NONE:
            return reads_tensor
        elif self.filter_type == FilterType.GAUSSIAN:
            return gaussian_smooth_1d(
                reads_tensor,
                kernel_size=self.filter_params.get('kernel_size', 5),
                sigma=self.filter_params.get('sigma', 1.0)
            )
        elif self.filter_type == FilterType.SAVGOL:
            return savgol_smooth_1d(
                reads_tensor,
                window_length=self.filter_params.get('window_length', 5),
                polyorder=self.filter_params.get('polyorder', 2)
            )
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

    @property
    def seqs(self):
        return Fasta(self.fasta_path)

    def apply_transform(self, count, max_count=None, min_count=None):
        """Apply the selected transform to the count value."""
        if self.transform_type == TransformType.NONE:
            return count
        elif self.transform_type == TransformType.LOG10:
            return np.log10(count) if count > 0 else 0
        elif self.transform_type == TransformType.MINMAX:
            return (count - min_count) / (max_count - min_count)
        elif self.transform_type == TransformType.LOG10_MINMAX:
            log_val = np.log10(count) if count > 0 else 0
            return (log_val - min_count) / (max_count - min_count)

    def __call__(self, chr_name, start, end, pileup_dir, return_augs=False):
        interval_length = end - start
        try:
            seqs = self.seqs
        except Exception:
            time.sleep(2)
            seqs = self.seqs
        try:
            chromosome = seqs[chr_name]
        except:
            chromosome = seqs[chr_name[3:]]
        chromosome_length = len(chromosome)

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

        # Initialize a column of zeros for the reads
        reads_tensor = torch.zeros((one_hot.shape[0], 1), dtype=torch.float)
        assert (
            reads_tensor.shape[0] == one_hot.shape[0]
        ), f"reads tensor must be same length as one hot tensor, reads: {reads_tensor.shape[0]} != one hot: {one_hot.shape[0]}"
        
        df = process_pileups(pileup_dir, chr_name, start, end)

        max_count = df["count"].max()
        min_count = df["count"].min()

        # Fill the reads tensor with scaled counts
        for row in df.iter_rows(named=True):
            position = row["position"]
            count = row["count"]
            relative_position = position - start - 1
            scaled_count = self.apply_transform(count, max_count, min_count)
            reads_tensor[relative_position, 0] = scaled_count

        # Apply selected smoothing filter
        smoothed_reads = self.apply_smoothing(reads_tensor)
        
        # Concatenate one-hot encoded sequence with smoothed reads
        extended_data = torch.cat((one_hot, smoothed_reads), dim=-1)

        if not return_augs:
            return extended_data

        rand_shift_tensor = torch.tensor([rand_shift])
        rand_aug_bool_tensor = torch.tensor([should_rc_aug])

        return extended_data, rand_shift_tensor, rand_aug_bool_tensor


class GenomeIntervalDataset(Dataset):
    def __init__(
        self,
        bed_file,
        fasta_file,
        cell_lines_dir,
        filter_df_fn=identity,
        chr_bed_to_fasta_map=dict(),
        context_length=None,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
        return_augs=False,
    ):
        super().__init__()

        # Initialization for GenomeIntervalDataset
        bed_path = Path(bed_file)
        assert bed_path.exists(), f"path to .bed file must exist: {bed_path}"

        df = pl.read_csv(str(bed_path), separator="\t", has_header=False)
        df = filter_df_fn(df)
        self.df = df

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

    def one_hot_encode_(self, labels):
        """
        One hot encodes the labels using the pre-initialized list of folders
        """
        labels_list = labels.split(",")

        labels_tensor = torch.zeros(len(CELL_LINES))

        for i, cell_line in enumerate(CELL_LINES):
            if cell_line in labels_list:
                labels_tensor[i] = 1
        return labels_tensor

    def __getitem__(self, ind):
        interval = self.df.row(ind)
        chr_name, start, end, cell_line, labels = (
            interval[0],
            interval[1],
            interval[2],
            interval[3],
            interval[4],
        )
        start, end = int(start), int(end)
        chr_name = self.chr_bed_to_fasta_map.get(chr_name, chr_name)

        labels_encoded = self.one_hot_encode_(labels)

        pileup_dir = self.cell_lines_dir / Path(cell_line) / "pileup/"
        # pileup_dir = self.cell_lines_dir / Path(cell_line) / "pileup_mod/"


        return (
            self.processor(
                chr_name, start, end, pileup_dir, return_augs=self.return_augs
            ),
            labels_encoded,
            ind
        )

    def __len__(self):
        return len(self.df)




def mask_sequence(input_tensor, mask_prob=0.15, mask_value=-1):
    """
    Masks the input sequence tensor with given probability.
    Masks the entire row including all columns.
    """
    # Clone the input tensor to create labels
    labels = input_tensor.clone()

    # Calculate row mask
    mask_rows = torch.bernoulli(
        torch.ones((input_tensor.shape[0], 1)) * mask_prob
    ).bool()

    # Expand mask to all columns
    mask = mask_rows.expand_as(input_tensor)

    # Apply mask to input_tensor to create the masked tensor
    masked_tensor = input_tensor.clone()
    masked_tensor[mask] = mask_value

    # Set the labels where mask is not True to -1 (or any invalid label)
    labels[~mask] = -1  # only calculate loss on masked tokens

    return masked_tensor, labels


class MaskedGenomeIntervalDataset(GenomeIntervalDataset):
    def __init__(self, mask_prob=0.15, *args, **kwargs):
        super(MaskedGenomeIntervalDataset, self).__init__(*args, **kwargs)
        self.mask_prob = mask_prob

    def __getitem__(self, index):
        seq, labels = super(MaskedGenomeIntervalDataset, self).__getitem__(index)

        # Mask the sequence and get the labels
        masked_seq, _ = mask_sequence(seq, mask_prob=self.mask_prob)

        return masked_seq, labels


if __name__ == "__main__":
    data_dir = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data"
    cell_lines_dir = "/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines"
    torch.manual_seed(42)



    dataset = GenomeIntervalDataset(
        bed_file=os.path.join("/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/utils/combined_peaks.bed"),
        fasta_file=os.path.join(data_dir, "genome.fa"),
        cell_lines_dir=cell_lines_dir,
        return_augs=False,
        rc_aug=False,
        shift_augs=(0, 0),
        context_length=16_384,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    for i, data in enumerate(dataloader):
        print(data)
        break
