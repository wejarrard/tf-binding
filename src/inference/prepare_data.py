import os
import tensorflow as tf
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.data_tf_weighted import TFIntervalDataset
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value is tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

data_dir = "training/data"
output_dir = "inference/data"
tfrecords_filename = os.path.join(output_dir, "dataset.tfrecords")

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

with tf.io.TFRecordWriter(tfrecords_filename) as writer:
    for item in tqdm(dataloader, desc="Saving TFRecords"):
        # Create a feature
        feature = {
            # Assuming the first three elements are tensors and need to be saved as byte arrays
            'tensor_0': _bytes_feature(tf.io.serialize_tensor(item[0])),
            'tensor_1': _bytes_feature(tf.io.serialize_tensor(item[1])),
            'tensor_2': _bytes_feature(tf.io.serialize_tensor(item[2])),
            'chr_name': _bytes_feature(item[3][0].encode()),  # Encode string to bytes
            'start': _int64_feature(int(item[4].item())),
            'end': _int64_feature(int(item[5].item())),
            'cell_line': _bytes_feature(item[6][0].encode()),
        }

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
