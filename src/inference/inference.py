# aws sagemaker-runtime invoke-endpoint \
#     --endpoint-name endpoint_name \
#     --body fileb://$file_name \
#     output_file.txt

import io
import json
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataloader import HDF5CSVLoader
import zipfile
from ..training.deepseq import DeepSeq


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tmpdir = "/tmp/"
tmpdir = "data"

def get_predictions(model, device: torch.device, val_loader: DataLoader):
    model.eval()

    result = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            inputs, targets, weights, chr_name, start, end, cell_line = (
                batch[0],
                batch[1],
                batch[2],
                batch[3],
                batch[4],
                batch[5],
                batch[6],
            )

            inputs, targets, weights = (
                inputs.to(device),
                targets.to(device),
                weights.to(device),
            )

            outputs = model(inputs)
            loss = F.binary_cross_entropy_with_logits(outputs, targets)

            loss_val = loss.item()

            # Calculate accuracy
            predicted = (outputs.data > 0.5).float()

            # Add to result

            result.append(
                [
                    chr_name,
                    start,
                    end,
                    cell_line,
                    targets.cpu().numpy(),
                    predicted.cpu().numpy(),
                    weights.cpu().numpy(),
                    loss_val,
                ]
            )

    result_df = pd.DataFrame(
        result,
        columns=[
            "chr_name",
            "start",
            "end",
            "cell_line",
            "targets",
            "predicted",
            "weights",
            "loss",
        ],
    )

    return result_df

def input_fn(request_body, request_content_type):
    # Ensure the content type is as expected
    assert request_content_type == "application/zip", f"Unexpected content type: {request_content_type}"
    
    # Assuming request_body is a binary stream of the zip file
    zip_path = os.path.join(tmpdir, "input.zip")
    
    # Write the incoming zip file to a temporary file
    with open(zip_path, "wb") as f:
        f.write(request_body)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tmpdir)
    
    # Assuming specific file names for hdf5 and csv, adjust as necessary
    hdf5_path = os.path.join(tmpdir, "tensors.hdf5")
    csv_path = os.path.join(tmpdir, "metadata.csv")
    
    # Validate that the expected files were extracted
    assert os.path.exists(hdf5_path), "HDF5 file not found in zip archive."
    assert os.path.exists(csv_path), "CSV file not found in zip archive."
    
    # Load the dataset
    dataset = HDF5CSVLoader(hdf5_path=hdf5_path, csv_path=csv_path)

    return dataset


def model_fn(model_dir):
    model = DeepSeq.from_hparams(
        dim=1536,
        depth=11,
        heads=8,
        target_length=-1,
        num_cell_lines=1,
        return_augs=True,
        num_downsamples=5,
    ).to(device)

    # model = transfer_enformer_weights_to_(model, transformer_only=True)
    state_dict = torch.load(
        os.path.join(model_dir, "pretrained_weights.pth"), map_location=device
    )
    modified_state_dict = {
        key.replace("_orig_mod.module.", ""): value for key, value in state_dict.items()
    }
    model.load_state_dict(modified_state_dict)

    for param in model.parameters():
        param.requires_grad = False

    model.to(device)

    return model


def predict_fn(input_object: Dataset, model: object):
    num_workers = 6

    dataloader = DataLoader(
        input_object,
        batch_size=8,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True
    )

    result_df = get_predictions(
        model=model,
        val_loader=dataloader,
        device=device,
    )

    return result_df


def output_fn(prediction, accept):
    """Serializes the prediction output to CSV if the requested content type is 'text/csv'."""
    assert accept == "text/csv", "Only 'text/csv' content type is supported."
    
    # Assuming 'prediction' is a DataFrame
    csv_buffer = io.StringIO()
    prediction.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)  # Rewind the buffer
    return csv_buffer.getvalue(), "text/csv"
