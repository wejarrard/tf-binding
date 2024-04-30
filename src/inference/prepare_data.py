import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import torch.multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.tf_dataloader import TFIntervalDataset

torch.multiprocessing.set_sharing_strategy("file_system")


def _torch_tensor_feature(value):
    """Returns a bytes_list from a string / byte."""
    value_byte = value.numpy().astype(np.float64).tobytes(order="C")
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value_byte]))


def _string_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def _string_feature_list(values):
    """Returns a bytes_list from a list of strings."""
    # Encode each string in the list into bytes and create a BytesList
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[v[0].encode() for v in values])
    )


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    value = int(value.item())
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def main() -> None:
    data_dir = "training/data"
    output_dir = "inference/data"
    base_tfrecords_filename = os.path.join(output_dir, "dataset")
    file_index = 1
    tfrecords_filename = f"{base_tfrecords_filename}_{file_index}.tfrecord"

    dataset = TFIntervalDataset(
        bed_file=os.path.join(data_dir, "AR_val"),
        fasta_file=os.path.join(data_dir, "genome.fa"),
        cell_lines_dir=os.path.join(data_dir, "cell_lines/"),
        return_augs=False,
        rc_aug=False,
        shift_augs=(0, 0),
        context_length=4_096,
        mode="inference",
        num_tfs=1,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    options = tf.io.TFRecordOptions(compression_type="GZIP")

    writer = tf.io.TFRecordWriter(tfrecords_filename, options)
    max_file_size = 99 * 1024 * 1024  # 99 MB in bytes

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
            # "tf_list": _string_feature_list(item[7]),
        }

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

        # Check if the current file size exceeds the limit
        if os.path.getsize(tfrecords_filename) > max_file_size:
            writer.close()  # Close the current file
            file_index += 1  # Increment the file index
            tfrecords_filename = (
                f"{base_tfrecords_filename}_{file_index}.tfrecord"  # New file name
            )
            writer = tf.io.TFRecordWriter(
                tfrecords_filename, options
            )  # Create a new writer

    writer.close()  # Ensure the last file is closed properly


if __name__ == "__main__":
    main()
