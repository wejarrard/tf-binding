import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import torch.multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.data_tf_weighted import TFIntervalDataset

torch.multiprocessing.set_sharing_strategy("file_system")


def _torch_tensor_feature(value):
    """Returns a bytes_list from a string / byte."""
    value_byte = value.numpy().astype(np.float64).tobytes(order="C")
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value_byte]))


def _string_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    value = int(value.item())
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def main() -> None:
    data_dir = "training/data"
    output_dir = "inference/data"
    tfrecords_filename = os.path.join(output_dir, "dataset.tfrecord")

    dataset = TFIntervalDataset(
        bed_file=os.path.join(data_dir, "AR_ATAC_broadPeak_val"),
        fasta_file=os.path.join(data_dir, "genome.fa"),
        cell_lines_dir=os.path.join(data_dir, "cell_lines/"),
        return_augs=False,
        rc_aug=False,
        shift_augs=(0, 0),
        context_length=16_384,
        mode="inference",
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    options = tf.io.TFRecordOptions(
        compression_type="GZIP",
        flush_mode=None,
        input_buffer_size=None,
        output_buffer_size=None,
        window_bits=None,
        compression_level=None,
        compression_method=None,
        mem_level=None,
        compression_strategy=None,
    )

    with tf.io.TFRecordWriter(tfrecords_filename, options) as writer:
        for item in tqdm(dataloader, desc="Preparing Data"):
            # Create a feature
            feature = {
                "input": _torch_tensor_feature(item[0]),
                "target": _torch_tensor_feature(item[1]),
                "weight": _torch_tensor_feature(item[2]),
                "chr_name": _string_feature(item[3][0]),
                "start": _int64_feature(item[4]),
                "end": _int64_feature(item[5]),
                "cell_line": _string_feature(item[6][0]),
            }

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())


if __name__ == "__main__":
    main()
