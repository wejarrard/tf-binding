#     --endpoint-name endpoint_name \
# aws sagemaker-runtime invoke-endpoint \
#     --body fileb://$file_name \
#     output_file.txt

import io
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
from tfrecord.torch.dataset import TFRecordDataset


from inference.deepseq import DeepSeq
from torchdata.datapipes.iter import FileLister, FileOpener, TFRecordLoader, Mapper


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



def load_model(model_dir):
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
        key.replace("_orig_mod.", ""): value for key, value in state_dict.items()
    }
    model.load_state_dict(modified_state_dict)

    for param in model.parameters():
        param.requires_grad = False

    model.to(device)

    return model


def predict(input_object: Dataset, model: object):
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


def recreate_tensor(binary_tensor, shape):
    return torch.from_numpy(np.frombuffer(str(binary_tensor), dtype=np.float64).reshape(shape))


def read_to_torch(item):
    """Convert 'input' binary data in a dictionary to a PyTorch tensor."""
    item['input'] = recreate_tensor(item['input'], [1, 16384, 5])
    # item['target'] = recreate_tensor(item['target'], [1])
    # item['weight'] = recreate_tensor(item['weight'], [1])

    # Return the item with the updated 'input'
    return item

def load_input():
    # Can also use https://pytorch.org/data/main/generated/torchdata.datapipes.iter.TFRecordLoader.html instead

    tfrecord_path = "./inference/data/dataset.tfrecord"

    index_path = None
    description = {
        'input': "byte",
        'target': "byte",
        'weight': "byte",
        'chr_name': "byte",
        'start': "int",
        'end': "int",
        'cell_line': "byte",
    }
    dataset = TFRecordDataset(tfrecord_path, index_path, description)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    return loader

def main():

    data = next(iter(load_input()))


    print(torch.from_numpy(np.frombuffer(data["input"], dtype=np.float64).reshape(1, 16384, 5)).shape)
    
if __name__ == "__main__":
    main()
