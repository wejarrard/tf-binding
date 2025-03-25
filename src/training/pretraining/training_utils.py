import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from enformer_pytorch import Enformer
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
import os
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler


def count_directories(path: str) -> int:
    # Check if path exists and is a directory
    assert os.path.exists(path), "The specified path does not exist."
    assert os.path.isdir(path), "The specified path is not a directory."

    # Count only directories within the specified path
    directory_count = sum(
        os.path.isdir(os.path.join(path, i)) for i in os.listdir(path)
    )
    return directory_count


def transfer_enformer_weights_to_(
    model: PreTrainedModel, transformer_only: bool = False
) -> PreTrainedModel:
    # Load pretrained weights
    enformer = Enformer.from_pretrained("EleutherAI/enformer-official-rough")

    # print(enformer)

    if transformer_only:
        # Specify components to transfer
        components_to_transfer = ["transformer"]

        # Initialize an empty state dict
        state_dict_to_load = {}

        # Iterate over each component
        for component in components_to_transfer:
            # Extract and add weights of the current component to the state dict
            component_dict = {
                key: value
                for key, value in enformer.state_dict().items()
                if key.startswith(component)
            }

            # Check if the component dict is not empty
            if component_dict:
                state_dict_to_load.update(component_dict)
                # Print confirmation if weights were transferred
                print(f"Weights successfully transferred for component: {component}")
            else:
                # Print a message if no weights were found for the component
                print(f"No weights to transfer for component: {component}")
    else:
        pass

    return model


def train_one_epoch(
    model,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    scaler: Optional[GradScaler] = None,
    scheduler: Optional[LRScheduler] = None,
) -> Tuple[float, float]:  # Returns both average loss and accuracy
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # Check if distributed training is initialized
    is_distributed = torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    for batch_idx, batch in enumerate(train_loader):
        inputs, targets, weights = batch[0], batch[1], batch[2]

        inputs, targets, weights = (
            inputs.to(device),
            targets.to(device),
            weights.to(device),
        )

        optimizer.zero_grad()

        # If scaler is provided, scale the loss
        if scaler and is_distributed:
            with autocast():
                outputs = model(inputs)
                loss = F.binary_cross_entropy_with_logits(
                    outputs, targets
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=0.2)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = F.binary_cross_entropy_with_logits(outputs, targets)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=0.2)

        # Step the optimizer
        optimizer.step()

        # Step the scheduler if provided
        if scheduler:
            scheduler.step()

        # Calculate and collect loss across all nodes
        if is_distributed:
            loss_tensor = torch.tensor([loss.item()], device=device)
            torch.distributed.all_reduce(loss_tensor)
            loss_val = loss_tensor.item() / torch.distributed.get_world_size()
        else:
            loss_val = loss.item()

        total_loss += loss_val

        # Calculate accuracy
        predicted = (outputs.data > 0.5).float()
        correct_predictions += (predicted == targets).sum().item()
        total_predictions += targets.numel()

        if rank == 0:

            if batch_idx % 1000 == 0:
                print(
                    f"Progress: {batch_idx}/{len(train_loader)} | Train Loss: {total_loss / (batch_idx + 1)} | LR: {scheduler.get_last_lr()[0]:.8f}"
                )

    average_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions * 100

    return average_loss, accuracy

def validate_one_epoch(
    model,
    criterion: nn.Module,
    device: torch.device,
    val_loader: DataLoader,
    n_classes=1,
    enable_umap=False,
    target_layer='linear_512',
    umap_samples=500,
    umap_output_dir='umap_results'
):
    """
    Validates one epoch and optionally performs UMAP visualization on intermediate activations.
    
    Args:
        model: The neural network model
        criterion: Loss function
        device: Device to run validation on
        val_loader: DataLoader for validation data
        n_classes: Number of output classes
        enable_umap: Whether to perform UMAP visualization
        target_layer: Name of the layer to extract activations from
        umap_samples: Maximum number of samples to use for UMAP
        umap_output_dir: Directory to save UMAP visualizations
    
    Returns:
        Tuple of (average_loss, accuracy)
    """

    model.eval()
    total_loss = 0.0
    
    # Check if distributed training is initialized
    is_distributed = torch.distributed.is_initialized()

    correct_predictions = torch.zeros(n_classes, device=device)
    total_predictions = torch.zeros(n_classes, device=device)
    
    # UMAP-related setup
    activations = {}
    all_activations = []
    all_labels = []
    sample_count = 0
    handle = None
    
    if enable_umap:
        # Find the target layer
        target_module = None
        for name, module in model.named_modules():
            if target_layer in name:
                target_module = module
                break
        
        if target_module is None:
            print(f"Warning: Layer {target_layer} not found. UMAP visualization will be skipped.")
            enable_umap = False
        else:
            # Define the hook function
            def hook_fn(module, input, output):
                activations[target_layer] = output.detach().cpu().numpy()
            
            # Register the hook
            handle = target_module.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            inputs, targets, weights = batch[0], batch[1], batch[2]

            inputs, targets, weights = (
                inputs.to(device),
                targets.to(device),
                weights.to(device),
            )

            outputs = model(inputs)
            loss = F.binary_cross_entropy_with_logits(outputs, targets, weight=weights)

            # Calculate and collect loss across all nodes
            if is_distributed:
                loss_tensor = torch.tensor([loss.item()], device=device)
                torch.distributed.all_reduce(loss_tensor)
                loss_val = loss_tensor.item() / torch.distributed.get_world_size()
            else:
                loss_val = loss.item()

            total_loss += loss_val

            # Calculate accuracy
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct = (predicted == targets).float()
            for i in range(n_classes):
                valid_targets = targets[:, i] != -1
                if valid_targets.any():
                    correct_predictions[i] += (
                        ((predicted[:, i] == targets[:, i]) & valid_targets)
                        .float()
                        .sum()
                    )
                    total_predictions[i] += valid_targets.sum()
            
            # Collect activations and labels for UMAP if enabled
            if enable_umap and target_layer in activations:
                batch_activations = activations[target_layer]
                batch_labels = targets.cpu().numpy()
                
                all_activations.append(batch_activations)
                all_labels.append(batch_labels)
                
                sample_count += batch_activations.shape[0]
                if sample_count >= umap_samples:
                    # We've collected enough samples
                    break
    
    # Remove the hook if it was registered
    if handle is not None:
        handle.remove()
    
    average_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions * 100

    # Process collected activations for UMAP if enabled
    if enable_umap and all_activations:
        try:
            
            # Create output directory if it doesn't exist
            os.makedirs(umap_output_dir, exist_ok=True)
            
            # Concatenate all collected data
            activations_array = np.vstack(all_activations)
            labels_array = np.vstack(all_labels)
            
            # Reshape if needed - flatten any dimensions beyond the first
            activations_array = activations_array.reshape(activations_array.shape[0], -1)
            
            print(f"Performing UMAP on {activations_array.shape[0]} samples with {activations_array.shape[1]} features")
            
            # Standardize the features
            scaler = StandardScaler()
            activations_scaled = scaler.fit_transform(activations_array)
            
            # Apply UMAP
            reducer = umap.UMAP(random_state=42)
            embedding = reducer.fit_transform(activations_scaled)
            
            # Plot for each class
            for i in range(n_classes):
                plt.figure(figsize=(10, 8))
                plt.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    c=labels_array[:, i],
                    cmap='viridis',
                    s=5,
                    alpha=0.8
                )
                plt.colorbar(label=f'Class {i} probability')
                plt.title(f'UMAP projection of {target_layer} activations (Class {i})')
                plt.savefig(os.path.join(umap_output_dir, f'umap_class_{i}.png'), dpi=300)
                plt.close()
            
            # Save the embeddings for further analysis
            np.save(os.path.join(umap_output_dir, 'umap_embeddings.npy'), embedding)
            np.save(os.path.join(umap_output_dir, 'labels.npy'), labels_array)
            
            print(f"UMAP visualizations saved to {umap_output_dir}")
            
        except ImportError as e:
            print(f"UMAP visualization failed: {e}")
            print("Please install required packages: pip install umap-learn matplotlib scikit-learn")
    
    return average_loss, accuracy