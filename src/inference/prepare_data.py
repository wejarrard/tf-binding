import os
import json
import gzip
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scripts.base_dataloader import TFIntervalDataset

def main(compression: str, max_file_size: int, data_dir="/Users/wejarrard/projects/tf-binding/data", output_path="/Users/wejarrard/projects/tf-binding/data/jsonl", input_file = "validation_combined.csv"):
    base_filename = os.path.join(output_path, "dataset")
    file_index = 1
    filename = f"{base_filename}_{file_index}.jsonl"

    if compression == 'gzip':
        filename += '.gz'
    dataset = TFIntervalDataset(
        bed_file=os.path.join(data_dir, 'data_splits', input_file),
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
                "motifs": item[7][0]
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
    parser.add_argument('--max_file_size', type=int, default=5, help="Specify maximum file size in MB")
    parser.add_argument('--data_dir', type=str, default="/Users/wejarrard/projects/tf-binding/data", help="Specify the output path")
    parser.add_argument('--output_path', type=str, default="/Users/wejarrard/projects/tf-binding/data/jsonl", help="Specify the output path")
    args = parser.parse_args()

    # remove files from data/jsonl
    for file in os.listdir(args.output_path):
        os.remove(os.path.join(args.output_path, file))

    main(args.compression, args.max_file_size, args.data_dir, args.output_path)
