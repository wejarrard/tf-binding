import io
import logging
import os
import sys
# from tempfile import NamedTemporaryFile

import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from deepseq import DeepSeq
from einops.layers.torch import Rearrange
from dataloader import JSONLinesDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def get_predictions(model, device: torch.device, val_loader):
    model.eval()
    
    # Initialize the activation dictionary
    activation = {}
    
    # Define and register the hook
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    hook_handle = model.out[1].register_forward_hook(get_activation('linear_512'))
    
    result = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            inputs, targets, weights, chr_name, start, end, cell_line, motifs, motif_score = (
                batch["input"],
                batch["target"],
                batch["weight"],
                batch["chr_name"],
                batch["start"],
                batch["end"],
                batch["cell_line"],
                batch['motifs'],
                batch['motif_score']
            )

            inputs, targets, weights = (
                inputs.to(device),
                targets.to(device),
                weights.to(device),
            )

            outputs = model(inputs)
            # Get sigmoid transformed outputs
            outputs = torch.sigmoid(outputs)

            # The activations are now stored in activation['linear_512']
            # Ensure that the activations correspond to the current batch
            linear_512_outputs = activation['linear_512']
            # If necessary, apply sigmoid or other transformations
            # linear_512_outputs = torch.sigmoid(linear_512_outputs)
            
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
                        targets[i].cpu().item(),
                        predicted[i].cpu().item(),
                        weights[i].cpu().item(),
                        outputs[i].cpu().item(),
                        linear_512_outputs[i].cpu().numpy(),
                        motifs[i],
                        motif_score[i]
                    ]
                )
            if (batch_idx + 1) % 50 == 0:
                logger.info(f"Processed {batch_idx + 1} batches.")
    
    # Remove the hook after we're done
    hook_handle.remove()
    
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
            "probabilities",
            "linear_512_output", 
            "motifs",
            "motif_score"
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
        num_downsamples=3,
    )

    model.out = nn.Sequential(
        nn.Linear(model.dim * 2, num_tfs),
        Rearrange("... c o -> ... o c"),
        nn.Linear(512, 1),
        nn.Flatten(),
    ).to(device)


    state_dict = torch.load(
        os.path.join(model_dir, "best_model.pth"), map_location=device
    )

    modified_state_dict = {
        key.replace("_orig_mod.", ""): value for key, value in state_dict.items()
    }
    model.load_state_dict(modified_state_dict)


    model.to(device)

    return model

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_dir, num_tfs=1)
    logger.info("Model loaded successfully.")
    return model.to(device)

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/jsonlines':
        logger.info("Reading JSONLines dataset")
        file_stream = io.BytesIO(request_body)
        dataset = JSONLinesDataset(file_stream=file_stream, num_tfs=1, compressed=True)
    else:
        raise ValueError(f"Unsupported content type or request body type: {request_content_type}, {type(request_body)}")
    
    return dataset

def predict_fn(input_object, model):
    dataloader = torch.utils.data.DataLoader(
        input_object,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True,
    )

    result_df = get_predictions(
        model=model,
        val_loader=dataloader,
        device=device,
    )

    return result_df

def output_fn(predictions, response_content_type):
    return predictions.to_json(orient="records")

if __name__ == "__main__":
    
    model = model_fn("/Users/wejarrard/projects/tf-binding/data/model")

    with open("/Users/wejarrard/projects/tf-binding/data/jsonl/dataset_1.jsonl.gz", "rb") as f:
        request_body = f.read()
    
    dataset = input_fn(request_body, "application/jsonlines")
    df = predict_fn(dataset, model)
    output = output_fn(df, "application/json")
    predictions_df = pd.read_json(output)
    accuracy = predictions_df['targets'].eq(predictions_df['predicted']).mean()
    logger.info(output)
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"False positive rate: {predictions_df[predictions_df['targets'] == 0]['predicted'].mean()}")
    logger.info(f"False negative rate: {predictions_df[predictions_df['targets'] == 1]['predicted'].mean()}")
