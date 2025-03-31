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

from data import CELL_LINES


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

        if batch is None:
            continue

        inputs, targets, weights = batch[0], batch[1], batch[2]

        print(inputs.shape, targets.shape, weights.shape)

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

            if batch_idx % 1_000 == 0:
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
    n_classes=len(CELL_LINES),
    enable_umap=False,
    target_layer='linear_512',
    umap_samples=500,
    umap_output_dir='umap_results'
):
    """
    Validates one epoch and performs UMAP visualization of cell lines in latent space.
    
    Args:
        model: The neural network model
        criterion: Loss function
        device: Device to run validation on
        val_loader: DataLoader for validation data
        n_classes: Number of output classes (cell lines)
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
    rank = torch.distributed.get_rank() if is_distributed else 0

    correct_predictions = torch.zeros(n_classes, device=device)
    total_predictions = torch.zeros(n_classes, device=device)
    
    # UMAP-related setup
    all_activations = []
    all_labels = []
    sample_count = 0
    activation = {}  # Dictionary to store activations
    
    if enable_umap and rank == 0:
        # Create output directory if it doesn't exist
        os.makedirs(umap_output_dir, exist_ok=True)
        
        # Define the hook function to capture activations
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().cpu()
            return hook
        
        # Find and register the hook based on target_layer
        hook_handle = None
        if target_layer == 'linear_512':
            # Assuming model.out[1] is the linear_512 layer
            try:
                hook_handle = model.out[1].register_forward_hook(get_activation(target_layer))
                print(f"Hook registered for layer: {target_layer}")
            except (AttributeError, IndexError) as e:
                print(f"Failed to register hook for {target_layer}: {e}")
                print("Attempting to find layer by iterating through model...")
                
                # Try to find the target layer by iterating
                for name, module in model.named_modules():
                    if target_layer in name or (hasattr(module, 'out_features') and 
                                              isinstance(module, nn.Linear) and 
                                              module.out_features == 512):
                        hook_handle = module.register_forward_hook(get_activation(target_layer))
                        print(f"Found and hooked layer: {name}")
                        break
        else:
            # Try to find the target layer by name
            for name, module in model.named_modules():
                if target_layer in name:
                    hook_handle = module.register_forward_hook(get_activation(target_layer))
                    print(f"Hook registered for layer: {name}")
                    break
        
        if hook_handle is None:
            print(f"Warning: Layer {target_layer} not found. UMAP visualization will be skipped.")
            enable_umap = False
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):

            if batch is None:
                continue

            inputs, targets, weights = batch[0], batch[1], batch[2]
            inputs, targets, weights = inputs.to(device), targets.to(device), weights.to(device)

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
            if enable_umap and target_layer in activation and rank == 0:
                batch_activations = activation[target_layer].numpy()
                batch_labels = targets.cpu().numpy()
                
                all_activations.append(batch_activations)
                all_labels.append(batch_labels)
                
                sample_count += batch_activations.shape[0]
                if sample_count >= umap_samples:
                    # We've collected enough samples
                    break
    
    # Remove the hook if it was registered
    if enable_umap and rank == 0 and hook_handle is not None:
        hook_handle.remove()
    
    # Calculate average loss and accuracy
    average_loss = total_loss / len(val_loader)
    
    # Handle the case where some classes have no predictions
    accuracy = torch.zeros_like(correct_predictions)
    valid_classes = total_predictions > 0
    accuracy[valid_classes] = correct_predictions[valid_classes] / total_predictions[valid_classes] * 100
    
    # Process collected activations for UMAP if enabled
    if enable_umap and all_activations and rank == 0:
        try:
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
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
            embedding = reducer.fit_transform(activations_scaled)
            
            # Save the embeddings for further analysis
            np.save(os.path.join(umap_output_dir, 'umap_embeddings.npy'), embedding)
            np.save(os.path.join(umap_output_dir, 'labels.npy'), labels_array)
            
            # Plot cell lines in the same UMAP space with different colors
            plt.figure(figsize=(12, 10))
            
            # Create a custom colormap for the cell lines
            # Minimum alpha threshold for visibility
            min_alpha = 0.5
            
            # Generate a color for each cell line
            colors = plt.cm.tab20(np.linspace(0, 1, len(CELL_LINES)))
            
            # Create a dictionary to keep track of class representations
            class_counts = {i: 0 for i in range(n_classes)}
            
            # Dominant class approach - assign each point to the cell line with highest probability
            dominant_classes = np.argmax(labels_array, axis=1)
            
            # Plot each point, colored by its dominant cell line
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                        c=dominant_classes, 
                        cmap=ListedColormap(colors),
                        s=15, alpha=0.8)
            
            # Create a legend
            legend_patches = []
            for i, cell_line in enumerate(CELL_LINES):
                # Only include cell lines that have some representation
                if np.any(dominant_classes == i):
                    class_counts[i] = np.sum(dominant_classes == i)
                    legend_patches.append(mpatches.Patch(
                        color=colors[i], 
                        label=f'{cell_line} (n={class_counts[i]})'
                    ))
            
            plt.legend(handles=legend_patches, loc='upper right', 
                      bbox_to_anchor=(1.15, 1), title="Cell Lines")
            
            plt.title('UMAP projection of cell lines in latent space', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(umap_output_dir, 'umap_cell_lines.png'), dpi=300, bbox_inches='tight')
            
            # Create an alternative visualization showing class probabilities
            plt.figure(figsize=(20, 16))
            
            # Create subplots for each cell line
            rows = int(np.ceil(n_classes / 4))
            fig, axes = plt.subplots(rows, 4, figsize=(20, 4 * rows))
            axes = axes.flatten()
            
            for i, cell_line in enumerate(CELL_LINES):
                if i < len(axes):  # Ensure we don't exceed the number of subplots
                    ax = axes[i]
                    
                    # Plot the probability for this cell line
                    scatter = ax.scatter(
                        embedding[:, 0], embedding[:, 1],
                        c=labels_array[:, i],
                        cmap='viridis',
                        s=10, alpha=0.7
                    )
                    
                    ax.set_title(f'{cell_line} (n={class_counts.get(i, 0)})')
                    fig.colorbar(scatter, ax=ax, label='Probability')
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            # Hide any unused subplots
            for i in range(n_classes, len(axes)):
                axes[i].axis('off')
                
            plt.tight_layout()
            plt.savefig(os.path.join(umap_output_dir, 'umap_cell_lines_probability.png'), 
                        dpi=300, bbox_inches='tight')
            
            # Create a heatmap of cell line correlation in the latent space
            plt.figure(figsize=(14, 12))
            
            # Calculate correlation between cell lines
            correlation_matrix = np.corrcoef(labels_array.T)
            
            # Plot correlation heatmap
            plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(label='Correlation')
            plt.title('Correlation between cell lines in latent space', fontsize=16)
            
            # Add cell line labels
            plt.xticks(range(len(CELL_LINES)), CELL_LINES, rotation=90)
            plt.yticks(range(len(CELL_LINES)), CELL_LINES)
            
            plt.tight_layout()
            plt.savefig(os.path.join(umap_output_dir, 'cell_line_correlation.png'), 
                       dpi=300, bbox_inches='tight')
            
            print(f"UMAP visualizations saved to {umap_output_dir}")
            
        except Exception as e:
            print(f"UMAP visualization failed: {e}")
            print("Make sure required packages are installed: pip install umap-learn matplotlib scikit-learn")
    
    return average_loss, accuracy.mean().item()