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

from sagemaker.s3 import S3Downloader
from sagemaker.session import Session

from ..training.data_tf_weighted import TFIntervalDataset
from ..training.deepseq import DeepSeq

sagemaker_session = Session()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def download_data_from_s3(s3_uri):
    S3Downloader.download(
        s3_uri=s3_uri, local_path="/tmp/", sagemaker_session=sagemaker_session
    )


def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    data = json.loads(request_body)["inputs"]
    data = torch.tensor(data, dtype=torch.float32, device=device)

    download_data_from_s3("s3://tf-binding-sites/pretraining/data/")

    dataset = TFIntervalDataset(
        bed_file=os.path.join("/tmp/", "AR_ATAC_broadPeak_train"),
        fasta_file=os.path.join("/tmp/", "genome.fa"),
        cell_lines_dir=os.path.join("/tmp/", "cell_lines/"),
        return_augs=False,
        rc_aug=False,
        shift_augs=(0, 0),
        context_length=16_384,
        mode="inference",
    )

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


def main(output_dir: str, data_dir: str) -> None:
    ############ DEVICE ############

    ############ DATA ############

    ############ TRAINING ############

    


if __name__ == "__main__":
    main(output_dir="/opt/ml/model", data_dir="/opt/ml/input/data")
