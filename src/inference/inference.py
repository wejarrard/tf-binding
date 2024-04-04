#     --endpoint-name endpoint_name \
# aws sagemaker-runtime invoke-endpoint \
#     --body fileb://$file_name \
#     output_file.txt

import os

import pandas as pd
import torch
from captum.attr import DeepLift
from dataloader import EnhancedTFRecordDataset
from deepseq import DeepSeq
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_predictions(model, device: torch.device, val_loader):
    model.eval()

    result = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            inputs, targets, weights, chr_name, start, end, cell_line, tf_list = (
                batch["input"],
                batch["target"],
                batch["weight"],
                batch["chr_name"],
                batch["start"],
                batch["end"],
                batch["cell_line"],
                batch["tf_list"],
            )

            inputs, targets, weights = (
                inputs.to(device),
                targets.to(device),
                weights.to(device),
            )

            baselines = torch.zeros_like(inputs)

            dl = DeepLift(model)

            attributions = dl.attribute(
                inputs=inputs,
                baselines=baselines,
                target=0,  # targets.squeeze().type(torch.int64),
                # return_convergence_delta=True,
            )
            outputs = model(inputs)
            # Get sigmoid transformed outputs
            outputs = torch.sigmoid(outputs)

            # Calculate accuracy
            predicted = (outputs.data > 0.5).float()

            # Add to result

            for i in range(predicted.shape[0]):
                result.append(
                    [
                        chr_name[i],
                        start[i].item(),
                        end[i].item(),
                        cell_line[i],
                        tf_list,
                        targets[i].cpu(),
                        predicted[i].cpu(),
                        weights[i].cpu(),
                        outputs[i].cpu(),
                        inputs,
                        attributions,
                    ]
                )
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {batch_idx + 1} batches.")

    result_df = pd.DataFrame(
        result,
        columns=[
            "chr_name",
            "start",
            "end",
            "cell_line",
            "tf_list",
            "targets",
            "predicted",
            "weights",
            "probabilities",
            "inputs",
            "attributions",
        ],
    )

    return result_df


def load_model(model_dir, num_tfs=1):
    model = DeepSeq.from_hparams(
        dim=1536,
        depth=11,
        heads=8,
        target_length=-1,
        num_cell_lines=num_tfs,
        return_augs=True,
        num_downsamples=5,
    ).to(device)

    # model = transfer_enformer_weights_to_(model, transformer_only=True)
    state_dict = torch.load(
        os.path.join(model_dir, "pretrained_weight.pth"), map_location=device
    )
    modified_state_dict = {
        key.replace("_orig_mod.", ""): value for key, value in state_dict.items()
    }
    model.load_state_dict(modified_state_dict)

    model.to(device)

    return model


def predict(input_object: Dataset, model: object):
    dataloader = DataLoader(
        input_object,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )

    result_df = get_predictions(
        model=model,
        val_loader=dataloader,
        device=device,
    )

    return result_df


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
    dataset = EnhancedTFRecordDataset(tfrecord_path, index_path, description)

    model_dir = "model"
    model = load_model(model_dir)

    result_df = predict(dataset, model)

    print(result_df)
