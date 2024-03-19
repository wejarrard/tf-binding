import os
import pandas as pd
import h5py
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.data_tf_weighted import TFIntervalDataset
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

data_dir = "training/data"
output_dir = "inference/data"

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

# Prepare HDF5 file for tensors
hdf5_path = os.path.join(output_dir, "tensors.hdf5")
with h5py.File(hdf5_path, 'w') as hdf_file:
    # Prepare CSV for metadata
    metadata = {'id': [], 'chr_name': [], 'start': [], 'end': [], 'cell_line': []}
    
    for i, item in enumerate(tqdm(dataloader, desc="Saving Data")):
        # Save tensors
        grp = hdf_file.create_group(str(i))
        for j, tensor in enumerate(item[:3]):  # Assuming the first three elements are tensors
            grp.create_dataset(name=f'tensor_{j}', data=tensor.numpy(), compression="gzip")
        
        # Save metadata, convert tensors and tuples to strings and ints
        metadata['id'].append(i)
        metadata['chr_name'].append(item[3][0])  # Assuming item[3] is a tuple like ('chr1',)
        metadata['start'].append(int(item[4].item()))  # Convert tensor to int
        metadata['end'].append(int(item[5].item()))  # Convert tensor to int
        metadata['cell_line'].append(item[6][0])  # Assuming item[6] is a tuple like ('LNCAP',)
    
    # Save metadata to CSV
    pd.DataFrame(metadata).to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
