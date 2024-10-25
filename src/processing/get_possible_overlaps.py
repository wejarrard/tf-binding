
import argparse
import os

parser = argparse.ArgumentParser(description="Process TF binding data.")

parser.add_argument("tf1", type=str, help="Transcription Factor 1")
parser.add_argument("tf2", type=str, help="Transcription Factor 2")

args = parser.parse_args()

tf1 = args.tf1
tf2 = args.tf2

tf1_all_dir = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{tf1}/merged"
tf2_all_dir = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{tf2}/merged"

tf1_processed_dir = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{tf1}/output"
tf2_processed_dir = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/scripts/data/transcription_factors/{tf2}/output"


def get_cell_lines(filename):
    return filename.split('_')[0]


tf1_all_cell_lines = [get_cell_lines(filename) for filename in os.listdir(tf1_all_dir)]
tf2_all_cell_lines = [get_cell_lines(filename) for filename in os.listdir(tf2_all_dir)]

tf1_processed_cell_lines = [get_cell_lines(filename) for filename in os.listdir(tf1_processed_dir)]
tf2_processed_cell_lines = [get_cell_lines(filename) for filename in os.listdir(tf2_processed_dir)]

print(f'Current overlap: {set(tf1_processed_cell_lines).intersection(set(tf2_processed_cell_lines))}')
print(f'Possible overlap: {set(tf1_all_cell_lines).intersection(set(tf2_all_cell_lines))}')
print(f"The remaining cell lines for {tf1} are: {set(tf1_all_cell_lines) - set(tf1_processed_cell_lines)}")
print(f"The remaining cell lines for {tf2} are: {set(tf2_all_cell_lines) - set(tf2_processed_cell_lines)}")
