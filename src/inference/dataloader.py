
import os
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset

class HDF5CSVLoader(Dataset):
    def __init__(self, hdf5_path, csv_path):
        self.hdf5_path = hdf5_path
        self.metadata = pd.read_csv(csv_path)
        self.hdf5_file = None  # To be opened in the context manager

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r')

        # Load tensor data from HDF5
        group = self.hdf5_file[str(idx)]
        tensors = [torch.tensor(group[f'tensor_{i}'][()]) for i in range(3)]  # Adjust if more tensors
        
        # Load metadata
        metadata_row = self.metadata.iloc[idx]
        chr_name = metadata_row['chr_name']
        start = metadata_row['start']
        end = metadata_row['end']
        cell_line = metadata_row['cell_line']
        
        # Example of processing the metadata, if necessary
        # Here you might need to adjust the format, especially if the tensors are expected in a specific shape

        return tensors + [chr_name, start, end, cell_line]

    def close(self):
        if self.hdf5_file is not None:
            self.hdf5_file.close()
