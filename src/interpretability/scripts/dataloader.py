from torch.utils.data import Dataset, DataLoader
import json
import gzip
import torch
from io import BytesIO, TextIOWrapper
import os

class JSONLinesDataset(Dataset):
    def __init__(self, file_stream, num_tfs=1, compressed=False):
        self.file_stream = file_stream
        self.num_tfs = num_tfs
        self.compressed = compressed
        self.data = self.load_data()

    def load_data(self):
        data = []
        if self.compressed:
            if isinstance(self.file_stream, (str, bytes, os.PathLike)):
                with gzip.open(self.file_stream, 'rt', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        data.append(self.process(item))
            elif isinstance(self.file_stream, BytesIO):
                with gzip.GzipFile(fileobj=self.file_stream) as gz:
                    f = TextIOWrapper(gz, encoding='utf-8')
                    for line in f:
                        item = json.loads(line)
                        data.append(self.process(item))
        else:
            if isinstance(self.file_stream, (str, bytes, os.PathLike)):
                with open(self.file_stream, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        data.append(self.process(item))
            elif isinstance(self.file_stream, BytesIO):
                self.file_stream.seek(0)  # Reset the stream position to the start
                f = TextIOWrapper(self.file_stream, encoding='utf-8')
                for line in f:
                    item = json.loads(line)
                    data.append(self.process(item))
        return data

    def recreate_tensor(self, data_list, shape):
        tensor = torch.tensor(data_list, dtype=torch.float32).reshape(shape)
        return tensor

    def process(self, item):
        item["input"] = self.recreate_tensor(item["input"], [4096, 5])
        item["target"] = self.recreate_tensor(item["target"], [self.num_tfs])
        item["weight"] = self.recreate_tensor(item["weight"], [self.num_tfs])
        item["motif_score"] = [float('-inf') if score is None else score for score in item["motif_score"]]
        return item

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



if __name__ == "__main__":
    jsonl_path = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/jsonl/AR_SRR12455436/dataset_1.jsonl.gz"  # Path to your JSON Lines file

    dataset = JSONLinesDataset(file_stream=jsonl_path, num_tfs=1, compressed=True)

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
        print(f"{key}: {value}")
