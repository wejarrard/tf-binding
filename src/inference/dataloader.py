import os

import numpy as np
import torch
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import DataLoader


class EnhancedTFRecordDataset(TFRecordDataset):
    def __init__(self, *args, **kwargs):
        super(EnhancedTFRecordDataset, self).__init__(*args, **kwargs)

    def __iter__(self):
        # Get the iterator from the parent class
        parent_iter = super(EnhancedTFRecordDataset, self).__iter__()

        # Use map to apply `process` to each item in the iterator
        processed_iter = map(self.process, parent_iter)

        # Yield from the processed iterator
        yield from processed_iter

    def recreate_tensor(self, binary_tensor, shape):
        tensor = torch.from_numpy(
            np.frombuffer(binary_tensor, dtype=np.float64)
            .copy()
            .reshape(shape)
            .astype(np.float32)
        )
        return tensor

    def process(self, item):
        """Convert 'input', 'target', and 'weight' binary data in a dictionary to PyTorch tensors."""
        item["input"] = self.recreate_tensor(item["input"], [16384, 5])
        item["target"] = self.recreate_tensor(item["target"], [1])
        item["weight"] = self.recreate_tensor(item["weight"], [1])

        item["chr_name"] = item["chr_name"].decode()
        item["cell_line"] = item["cell_line"].decode()

        # Return the item with the updated fields
        return item


if __name__ == "__main__":

    tfrecord_path = "./data/dataset.tfrecord"

    index_path = None
    description = {
        "input": "byte",
        "target": "byte",
        "weight": "byte",
        "chr_name": "byte",
        "start": "int",
        "end": "int",
        "cell_line": "byte",
    }
    dataset = EnhancedTFRecordDataset(
        tfrecord_path, index_path, description, compression_type="gzip"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )

    data = next(iter(dataloader))

    for key, value in data.items():
        print(type(value))
