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
import torch
from typing import Tuple, Optional
import torch.nn.functional as F
from captum.attr import DeepLift


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))



def gaussian_kernel1d(sigma: float, kernel_size: int) -> torch.Tensor:
    """Create 1D Gaussian kernel"""
    x = torch.linspace(-3*sigma, 3*sigma, kernel_size)
    kernel = torch.exp(-0.5 * (x/sigma)**2)
    return kernel / kernel.sum()


class BaselineGenerator:
    def __init__(self, 
                 seq_length: int,
                 n_channels: int = 5,
                 device: str = 'cpu'):
        self.seq_length = seq_length
        self.n_channels = n_channels
        self.device = device
        
    def dinuc_shuffle_sequence(self, 
                             one_hot_seq: torch.Tensor) -> torch.Tensor:
        """Shuffle sequence while preserving dinucleotide frequencies"""
        # Convert one-hot back to sequence
        seq = torch.argmax(one_hot_seq, dim=1)
        
        # Get dinucleotide edges
        edges = list(zip(seq[:-1].tolist(), seq[1:].tolist()))
        
        # Eulerian shuffle implementation
        shuffled = seq.tolist()
        for i in range(len(seq) - 2):
            j = torch.randint(i + 1, len(seq) - 1, (1,)).item()
            if (shuffled[i], shuffled[i + 1]) in edges and \
               (shuffled[j], shuffled[j + 1]) in edges:
                shuffled[i + 1], shuffled[j] = shuffled[j], shuffled[i + 1]
                
        # Convert back to one-hot
        shuffled = torch.tensor(shuffled, device=self.device)
        shuffled_one_hot = torch.zeros_like(one_hot_seq)
        shuffled_one_hot[torch.arange(len(shuffled), device=self.device), shuffled] = 1
        return shuffled_one_hot

    def smooth_signal1d(self, 
                       signal: torch.Tensor,
                       sigma: float,
                       kernel_size: int) -> torch.Tensor:
        """Apply 1D Gaussian smoothing"""
        kernel = gaussian_kernel1d(sigma, kernel_size).to(signal.device)
        # Pad signal to handle edges
        pad_size = kernel_size // 2
        padded = F.pad(signal.view(1, 1, -1), (pad_size, pad_size), mode='reflect')
        # Apply convolution
        smoothed = F.conv1d(padded, kernel.view(1, 1, -1), padding=0)
        return smoothed.view(-1)

    def generate_atac_baselines(self, 
                              atac_signal: torch.Tensor,
                              method: str = 'mean',
                              smooth_sigma: Optional[float] = 2.0) -> torch.Tensor:
        """Generate baseline for ATAC channel"""
        if method == 'mean':
            # Use mean signal as baseline
            return torch.full_like(atac_signal, atac_signal.mean())
            
        elif method == 'shuffle':
            # Shuffle while preserving distribution and local structure
            shuffled = atac_signal[torch.randperm(len(atac_signal))]
            if smooth_sigma is not None:
                # Convert sigma to kernel size (6*sigma rounded to nearest odd)
                kernel_size = int(6 * smooth_sigma) // 2 * 2 + 1
                shuffled = self.smooth_signal1d(shuffled, smooth_sigma, kernel_size)
            # Normalize to match original mean and std
            shuffled = (shuffled - shuffled.mean()) / shuffled.std()
            shuffled = shuffled * atac_signal.std() + atac_signal.mean()
            return shuffled
            
        elif method == 'sample':
            # Sample from empirical distribution
            indices = torch.randint(0, len(atac_signal), (len(atac_signal),), device=self.device)
            sampled = atac_signal[indices]
            if smooth_sigma is not None:
                kernel_size = int(6 * smooth_sigma) // 2 * 2 + 1
                sampled = self.smooth_signal1d(sampled, smooth_sigma, kernel_size)
            return sampled

        elif method == 'random':
            # Generate random values between -1 and 1
            return torch.rand_like(atac_signal) * 2 - 1
            
        else:
            raise ValueError(f"Unknown method: {method}")

    def generate_baselines(self, 
                          input_data: torch.Tensor,
                          atac_method: str = 'mean',
                          smooth_sigma: Optional[float] = 2.0) -> torch.Tensor:
        """Generate baseline for full input tensor"""
        baselines = torch.zeros_like(input_data)
        
        # Handle sequence channels (0-3)
        seq_channels = input_data[..., :4]
        for i in range(len(input_data)):
            baselines[i, :, :4] = self.dinuc_shuffle_sequence(seq_channels[i])
        
        # Handle ATAC channel (4)
        atac_channel = input_data[..., 4]
        for i in range(len(input_data)):
            baselines[i, :, 4] = self.generate_atac_baselines(
                atac_channel[i],
                method=atac_method,
                smooth_sigma=smooth_sigma
            )
            
        return baselines


def map_state_dict_to_unique_activations(old_state_dict):
    """
    Maps the old state dictionary to work with our new model architecture that uses unique activations.
    
    This function creates a new state dictionary that maintains all the original weights and parameters
    while being compatible with our new unique activation function structure.
    
    Args:
        old_state_dict (OrderedDict): The original model's state dictionary
        
    Returns:
        OrderedDict: A new state dictionary compatible with our modified architecture
    """
    new_state_dict = {}
    
    # Create a mapping of old keys to new keys
    key_mapping = {}
    
    # Iterate through old state dict keys
    for key in old_state_dict.keys():
        # The key structure usually looks like 'layer_name.sublayer.weight'
        new_key = key
        
        # Check if this is an activation function parameter
        if 'relu' in key.lower() or 'gelu' in key.lower():
            # Extract the layer path to create a unique identifier
            path_parts = key.split('.')
            layer_path = '_'.join(path_parts[:-1])
            new_key = f"{layer_path}.unique_{path_parts[-1]}"
        
        key_mapping[key] = new_key
        new_state_dict[new_key] = old_state_dict[key]
    
    return new_state_dict


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

    dl = DeepLift(model)
    
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
            
            generator = BaselineGenerator(
                seq_length=inputs.shape[1],
                n_channels=inputs.shape[2],
                device=device,
            )
            
            baselines = generator.generate_baselines(
                inputs,
                atac_method="mean",
            )


            attributions = dl.attribute(inputs, baselines=baselines)

            # The activations are now stored in activation['linear_512']
            # Ensure that the activations correspond to the current batch
            linear_512_outputs = activation['linear_512']
            # If necessary, apply sigmoid or other transformations
            # linear_512_outputs = torch.sigmoid(lin ear_512_outputs)
            
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
                        attributions[i].cpu().numpy(),
                        # motif_score[i].item()
                    ]
                )
            # if (batch_idx + 1) % 50 == 0:
            #     logger.info(f"Processed {batch_idx + 1} batches.")
    
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
            "attributions",
            # "motif_score"
        ],
    )

    return result_df

def load_model(model_dir: str, num_tfs=1):

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
        os.path.join(model_dir, "best_model.pth"), map_location=device, weights_only=True
    )

    intermediate_state_dict = map_state_dict_to_unique_activations(state_dict)

    modified_state_dict = {
        key.replace("_orig_mod.", ""): value 
        for key, value in intermediate_state_dict.items()
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

    root_dir = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"
    
    model = model_fn(os.path.join(root_dir, "data/models"))

    with open(os.path.join(root_dir, "data/jsonl/AR_22Rv1/dataset_1.jsonl.gz"), "rb") as f:
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
