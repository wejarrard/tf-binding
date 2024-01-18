import os
from tabnanny import check
from typing import Optional, Tuple

import torch
import torch.nn as nn
from enformer_pytorch import Enformer
from torch.cuda.amp.autocast_mode import autocast
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from utils.checkpointing import save_checkpoint
from utils.protocols import ScalerProtocol, SchedulerProtocol


def get_params_without_weight_decay_ln(named_params, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in named_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in named_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


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
    epoch: int,
    hyperparams,
    early_stopping,
    scaler: Optional[ScalerProtocol] = None,
    scheduler: Optional[SchedulerProtocol] = None,
    current_batch: int = 0,
    max_grad_norm: float = 0.2,
    log_frequency: int = 5000,
    checkpointing=True,
) -> Tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # Check if distributed training is initialized
    is_distributed = torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx < current_batch:
            continue
        inputs, targets, data_inds = batch[0], batch[1], batch[2]
        assert not torch.isnan(inputs).any(), f"NaNs in inputs {data_inds}"
        assert not torch.isnan(targets).any(), f"NaNs in targets {data_inds}"

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # If scaler is provided, scale the loss
        if scaler and is_distributed:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

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
            if batch_idx % log_frequency == 0 and batch_idx != 0:
                print(
                    f"Progress: {batch_idx}/{len(train_loader)} | Train Loss: {total_loss / (batch_idx + 1)} |"
                    + f"LR: {scheduler.get_last_lr()[0]:.8f}"
                    if scheduler
                    else ""
                )
                if checkpointing:
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        early_stopping,
                        epoch,
                        hyperparams,
                        batch_idx,
                    )

    average_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions * 100

    return average_loss, accuracy


def validate_one_epoch(
    model,
    criterion: nn.Module,
    device: torch.device,
    val_loader: DataLoader,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # Check if distributed training is initialized
    is_distributed = torch.distributed.is_initialized()

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            inputs, targets, data_inds = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2],
            )

            outputs = model(inputs)
            loss = criterion(outputs, targets)

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

    average_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions * 100

    return average_loss, accuracy
import os
from tabnanny import check
from typing import Optional, Tuple

import torch
import torch.nn as nn
from enformer_pytorch import Enformer
from torch.cuda.amp.autocast_mode import autocast
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from utils.checkpointing import save_checkpoint
from utils.protocols import ScalerProtocol, SchedulerProtocol


def get_params_without_weight_decay_ln(named_params, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in named_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in named_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


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
    epoch: int,
    hyperparams,
    early_stopping,
    scaler: Optional[ScalerProtocol] = None,
    scheduler: Optional[SchedulerProtocol] = None,
    current_batch: int = 0,
    max_grad_norm: float = 0.2,
    log_frequency: int = 5000,
    checkpointing=True,
) -> Tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # Check if distributed training is initialized
    is_distributed = torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx < current_batch:
            continue
        inputs, targets, data_inds = batch[0], batch[1], batch[2]
        assert not torch.isnan(inputs).any(), f"NaNs in inputs {data_inds}"
        assert not torch.isnan(targets).any(), f"NaNs in targets {data_inds}"

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # If scaler is provided, scale the loss
        if scaler and is_distributed:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

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
            if batch_idx % log_frequency == 0 and batch_idx != 0:
                print(
                    f"Progress: {batch_idx}/{len(train_loader)} | Train Loss: {total_loss / (batch_idx + 1)} |"
                    + f"LR: {scheduler.get_last_lr()[0]:.8f}"
                    if scheduler
                    else ""
                )
                if checkpointing:
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        early_stopping,
                        epoch,
                        hyperparams,
                        batch_idx,
                    )

    average_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions * 100

    return average_loss, accuracy


def validate_one_epoch(
    model,
    criterion: nn.Module,
    device: torch.device,
    val_loader: DataLoader,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # Check if distributed training is initialized
    is_distributed = torch.distributed.is_initialized()

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            inputs, targets, data_inds = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2],
            )

            outputs = model(inputs)
            loss = criterion(outputs, targets)

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

    average_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions * 100

    return average_loss, accuracy
