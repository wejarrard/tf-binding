import os
import json
import gzip
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scripts.base_dataloader import TFIntervalDataset

def main(compression: str, max_file_size: int) -> None:
    data_dir = "/Users/wejarrard/projects/tf-binding/data"
    output_dir = "/Users/wejarrard/projects/tf-binding/data/jsonl"
    base_filename = os.path.join(output_dir, "dataset")
    file_index = 1
    filename = f"{base_filename}_{file_index}.jsonl"

    if compression == 'gzip':
        filename += '.gz'

    dataset = TFIntervalDataset(
        bed_file=os.path.join(data_dir, "validation_THP-1.csv"),
        fasta_file=os.path.join(data_dir, "genome.fa"),
        cell_lines_dir=os.path.join(data_dir, "cell_lines/"),
        return_augs=False,
        rc_aug=False,
        shift_augs=(0, 0),
        context_length=4_096,
        mode="inference",
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    max_file_size_bytes = max_file_size * 1024 * 1024  # Convert MB to bytes

    def open_file(filename, mode='wt', encoding='utf-8'):
        if compression == 'gzip':
            return gzip.open(filename, mode, encoding=encoding)
        else:
            return open(filename, mode, encoding=encoding)

    with open_file(filename) as jsonl_file:
        for item in tqdm(dataloader, desc="Preparing Data"):
            data_point = {
                "input": item[0].numpy().tolist(),
                "target": item[1].numpy().tolist(),
                "weight": item[2].numpy().tolist(),
                "chr_name": item[3][0],
                "start": item[4].item(),
                "end": item[5].item(),
                "cell_line": item[6][0],
            }
            jsonl_file.write(json.dumps(data_point) + '\n')

            # Check if the current file size exceeds the limit
            if os.path.getsize(filename) > max_file_size_bytes:
                jsonl_file.close()
                file_index += 1  # Increment the file index
                filename = f"{base_filename}_{file_index}.jsonl"
                if compression == 'gzip':
                    filename += '.gz'
                jsonl_file = open_file(filename)

        jsonl_file.close()  # Ensure the last file is closed properly

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for TF binding.")
    parser.add_argument('--compression', choices=['gzip', 'none'], default='gzip', help="Specify compression type (gzip or none)")
    parser.add_argument('--max_file_size', type=int, default=99, help="Specify maximum file size in MB")
    args = parser.parse_args()
    main(args.compression, args.max_file_size)
