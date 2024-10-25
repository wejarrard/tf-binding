import os
import orjson
import gzip
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scripts.base_dataloader import TFIntervalDataset
import shutil  # Moved import to the top for better practice

torch.multiprocessing.set_sharing_strategy('file_system')

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
        if compression == 'gzip':
            return gzip.open(filename, mode)
        else:
            return open(filename, mode)

    # Initialize the first file
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
                "motifs": item[7][0]
            }
            # Serialize using orjson and append a newline byte
            json_bytes = orjson.dumps(data_point) + b'\n'
            jsonl_file.write(json_bytes)

            # Check if the current file size exceeds the limit
            if os.path.getsize(filename) > max_file_size_bytes:
                jsonl_file.close()
                file_index += 1  # Increment the file index
                filename = f"{base_filename}_{file_index}.jsonl"
                if compression == 'gzip':
                    filename += '.gz'
                jsonl_file = open_file(filename)
    finally:
        # Ensure the last file is closed properly even if an error occurs
        jsonl_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for TF binding.")

    # Compression and File Size Configurations
    parser.add_argument(
        '--compression',
        choices=['gzip', 'none'],
        default='gzip',
        help="Specify compression type (gzip or none) (default: 'gzip')"
    )
    parser.add_argument(
        '--max_file_size',
        type=int,
        default=5,
        help="Specify maximum file size in MB (default: 5)"
    )

    # Directory and File Configurations
    parser.add_argument(
        '--data_dir',
        type=str,
        default="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data",
        help='Specify the data directory (default: "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data")'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/jsonl",
        help='Specify the output path (default: "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/jsonl")'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default="validation_combined.csv",
        help='Specify the input CSV file name (default: "validation_combined.csv")'
    )
    parser.add_argument(
        '--cell_line_dir',
        type=str,
        default="/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines",
        help='Specify the cell line directory (default: "/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines")'
    )

    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Remove existing files from the output directory
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