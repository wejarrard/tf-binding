import ast
import os
import time
from pathlib import Path
from random import random, randrange

import numpy as np
import polars as pl
import pysam
import torch
import torch.nn.functional as F
from einops import rearrange
from pyfaidx import Fasta
from torch.utils.data import DataLoader, Dataset

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
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), "path to fasta file must exist"
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
        reads_tensor = torch.zeros((one_hot.shape[0], 1), dtype=torch.long)
        assert (
            reads_tensor.shape[0] == one_hot.shape[0]
        ), f"reads tensor must be same length as one hot tensor, reads: {reads_tensor.shape[0]} != one hot: {one_hot.shape[0]}"
        extended_data = torch.cat((one_hot, reads_tensor), dim=-1)

        df = process_pileups(pileup_dir, chr_name, start, end)

        # max_count = df["count"].max() if df.height > 0 else 1

        # Iterate over the rows of the filtered DataFrame and update the reads_tensor with count data
        for row in df.iter_rows(named=True):
            position = row["position"]
            # count = int(10 ** row["count"])  # Reverse the log base 10 transformation
            count = np.log10(row["count"]) if row["count"] > 0 else 0

            # Calculate the relative position directly without using a separate position_tensor
            relative_position = position - start - 1

            # ADD IN IF YOU WANT TO STANDARDIZE, MUST ALSO CHANGE EXTENDED DATA DIRECTLY BELOW THIS
            # standardized_count = count / max_count if max_count else 0

            # Update the respective position in the extended_data tensor
            extended_data[relative_position, 4] = count

        if not return_augs:
            return extended_data
            # return one_hot

        rand_shift_tensor = torch.tensor([rand_shift])
        rand_aug_bool_tensor = torch.tensor([should_rc_aug])

        return extended_data, rand_shift_tensor, rand_aug_bool_tensor


class TFIntervalDataset(Dataset):
    def __init__(
        self,
        bed_file,
        fasta_file,
        cell_lines_dir,
        filter_df_fn=identity,
        chr_bed_to_fasta_map=dict(),
        mode="train",
        num_tfs=2,
        context_length=None,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
        return_augs=False,
    ):
        super().__init__()

        # Initialization for GenomeIntervalDataset
        bed_path = Path(bed_file)
        assert bed_path.exists(), "path to .bed file must exist"

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
        self.label_folders = sorted(
            [f.name for f in self.cell_lines_dir.iterdir() if f.is_dir()],
            key=lambda x: x,
        )
        self.mode = mode

    def process_tfs(self, score: int, label: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([score]), torch.tensor([label], dtype=torch.float32)


    def __getitem__(self, ind):
        interval = self.df.row(ind)
        chr_name, start, end, score, label, cell_line, motifs = (
            interval[0],
            interval[1],
            interval[2],
            interval[3],
            interval[4],
            interval[5],
            interval[6],
        )
        chr_name = self.chr_bed_to_fasta_map.get(chr_name, chr_name)

        score, label_encoded = self.process_tfs(score, label)

        # pileup_dir = self.cell_lines_dir / Path(cell_line)
        # pileup_dir = self.cell_lines_dir / "mod_log10" / Path(cell_line)
        pileup_dir = self.cell_lines_dir / Path(cell_line) / "pileup_mod"
        if self.mode == "train":
            return (
                self.processor(
                    chr_name, start, end, pileup_dir, return_augs=self.return_augs
                ),
                label_encoded,
                score,
            )
        elif self.mode == "inference":
            return (
                self.processor(
                    chr_name, start, end, pileup_dir, return_augs=self.return_augs
                ),
                label_encoded,
                score,
                chr_name,
                start,
                end,
                cell_line,
                motifs
            )
        else:
            return (
                self.processor(
                    chr_name, start, end, pileup_dir, return_augs=self.return_augs
                )
            )

    def __len__(self):
        return len(self.df)

if __name__ == "__main__":
    data_dir = "/Users/wejarrard/projects/tf-binding/data"
    train_dataset = TFIntervalDataset(
        bed_file=os.path.join(data_dir, "22Rv1.csv"),
        fasta_file=os.path.join(data_dir, "genome.fa"),
        cell_lines_dir=os.path.join(data_dir, "cell_lines/"),
        return_augs=False,
        rc_aug=True,
        shift_augs=(-50, 50),
        context_length=4_096,
        mode="inference",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    for i, data in enumerate(train_loader):
        inputs, labels, scores, chr_names, starts, ends, cell_lines = data
        print(scores)
