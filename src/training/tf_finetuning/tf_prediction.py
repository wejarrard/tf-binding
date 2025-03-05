import argparse
import os
import warnings
from dataclasses import dataclass
from enum import Enum
from multiprocessing import cpu_count
from typing import Tuple, Optional

import pysam
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch._dynamo import optimize
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from einops.layers.torch import Rearrange

from deepseq import DeepSeq
from earlystopping import EarlyStopping
from tf_dataloader import TFIntervalDataset, TransformType, FilterType, Mode


# ---------------------------------------------
# Constants
# ---------------------------------------------
SEED_VALUE: int = 42
BATCH_SIZE_FALLBACK: int = 1
MAX_GRAD_NORM: float = 0.2
WARMUP_STEPS_GPU: int = 1000
WARMUP_STEPS_CPU: int = 0
PRINT_FREQ: int = 1000
DEFAULT_CONTEXT_LENGTH: int = 4096
DEFAULT_LR: float = 5e-4
DEFAULT_WEIGHT_DECAY: float = 0.1
DEFAULT_BETAS: Tuple[float, float] = (0.9, 0.999)
DEFAULT_EPS: float = 1e-8
CHECK_LOSS_NAN: bool = True

# ---------------------------------------------
# Enums
# ---------------------------------------------
class RunMode(Enum):
    DISTRIBUTED = "distributed"
    STANDALONE = "standalone"


# ---------------------------------------------
# Hyperparameter configuration
# ---------------------------------------------
@dataclass
class HyperParams:
    num_epochs: int = 50
    batch_size: int = 16 if torch.cuda.is_available() else BATCH_SIZE_FALLBACK
    learning_rate: float = DEFAULT_LR
    early_stopping_patience: int = 2
    focal_loss_alpha: float = 1.0
    focal_loss_gamma: float = 2.0
    transform_type: TransformType = TransformType.NONE
    filter_type: FilterType = FilterType.NONE


# ---------------------------------------------
# Utility functions
# ---------------------------------------------
def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def suppress_warnings() -> None:
    """Suppress irrelevant warnings for cleaner logs."""
    warnings.filterwarnings("ignore", category=UserWarning)
    pysam.set_verbosity(0)


def determine_run_mode() -> RunMode:
    """Determine if training should run in distributed or standalone mode."""
    sm_hosts_str: str = os.environ.get("SM_HOSTS", "")
    sm_hosts = [h for h in sm_hosts_str.split(",") if h]
    if len(sm_hosts) > 1:
        return RunMode.DISTRIBUTED
    if torch.cuda.device_count() > 1:
        return RunMode.DISTRIBUTED
    return RunMode.STANDALONE


def configure_distributed_mode(local_rank: int) -> torch.device:
    """
    Configure process group and device for distributed mode.
    Exits if initialization fails.
    """
    try:
        dist.init_process_group(backend="nccl")
    except Exception as e:
        raise RuntimeError("Failed to init process group.") from e
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return device


def get_device(run_mode: RunMode, local_rank: int) -> torch.device:
    """Get the device to run on based on run mode and GPU availability."""
    if torch.cuda.is_available():
        if run_mode == RunMode.DISTRIBUTED:
            return configure_distributed_mode(local_rank)
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(device: torch.device, data_dir: str) -> nn.Module:
    """Build and load the model with pretrained weights."""
    num_cell_lines = 33
    pretrained_model = DeepSeq.from_hparams(
        dim=1536,
        depth=11,
        heads=8,
        target_length=-1,
        num_cell_lines=num_cell_lines,
        return_augs=True,
        num_downsamples=5,
    ).to(device)

    state_dict_path = os.path.join(data_dir, "pretrained_weights.pth")
    if not os.path.isfile(state_dict_path):
        raise FileNotFoundError(f"Pretrained weights not found at {state_dict_path}")

    state_dict = torch.load(state_dict_path, map_location=device)
    modified_state_dict = {
        key.replace("_orig_mod.module.", ""): value for key, value in state_dict.items()
    }
    pretrained_model.load_state_dict(modified_state_dict)

    updated_config = pretrained_model.config
    updated_config.num_downsamples = 3

    # Initialize the smaller dimension model
    model = DeepSeq(updated_config)

    # Transfer weights
    model.stem.load_state_dict(pretrained_model.stem.state_dict())
    model.transformer.load_state_dict(pretrained_model.transformer.state_dict())
    model.final_pointwise.load_state_dict(pretrained_model.final_pointwise.state_dict())
    model.out.load_state_dict(pretrained_model.out.state_dict())

    # Customize output head
    model.out = nn.Sequential(
        nn.Linear(model.dim * 2, 1),
        Rearrange("... c o -> ... o c"),
        nn.Linear(512, 1),
        nn.Flatten(),
    )

    for param in model.parameters():
        param.requires_grad = True

    del pretrained_model, modified_state_dict, state_dict

    return model


def get_params_without_weight_decay_ln(named_params, weight_decay: float):
    """Group parameters into those with and without weight decay."""
    no_decay_keys = ["bias", "LayerNorm.weight"]
    decay_params = [p for n, p in named_params if not any(nd in n for nd in no_decay_keys)]
    no_decay_params = [p for n, p in named_params if any(nd in n for nd in no_decay_keys)]

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def get_data_loaders(
    data_dir: str,
    train_file: str,
    valid_file: str,
    batch_size: int,
    num_workers: int
) -> Tuple[DataLoader, DataLoader]:
    """Prepare and return training and validation data loaders."""
    train_dataset = TFIntervalDataset(
        bed_file=os.path.join(data_dir, train_file),
        fasta_file=os.path.join(data_dir, "genome.fa"),
        cell_lines_dir=os.path.join(data_dir, "cell_lines/"),
        return_augs=False,
        rc_aug=True, # Flip or not flip the sequence
        shift_augs=(-500, 500),
        context_length=DEFAULT_CONTEXT_LENGTH,
        mode=Mode.TRAIN,
        transform_type=hyperparams.transform_type,
        filter_type=hyperparams.filter_type,
    )

    valid_dataset = TFIntervalDataset(
        bed_file=os.path.join(data_dir, valid_file),
        fasta_file=os.path.join(data_dir, "genome.fa"),
        cell_lines_dir=os.path.join(data_dir, "cell_lines/"),
        return_augs=False,
        rc_aug=False,
        context_length=DEFAULT_CONTEXT_LENGTH,
        mode=Mode.VALIDATION,
        transform_type=hyperparams.transform_type,
        filter_type=hyperparams.filter_type,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, valid_loader


def run_train_step(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device
) -> float:
    """Run a single training step and return the loss."""
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = F.binary_cross_entropy_with_logits(outputs, targets)
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
    optimizer.step()
    if scheduler:
        scheduler.step()
    return loss.item()


def run_train_step_distributed(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler
) -> float:
    """Run a single training step under distributed settings, using scaling."""
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss = F.binary_cross_entropy_with_logits(outputs, targets)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
    scaler.step(optimizer)
    scaler.update()
    if scheduler:
        scheduler.step()
    return loss.item()


def aggregate_loss(loss_val: float, device: torch.device) -> float:
    """Aggregate loss across distributed workers."""
    loss_tensor = torch.tensor([loss_val], device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    world_size = dist.get_world_size()
    return (loss_tensor.item() / world_size)


def compute_accuracy(outputs: Tensor, targets: Tensor) -> float:
    """Compute accuracy given model outputs and targets."""
    predicted = (torch.sigmoid(outputs) > 0.5).float()
    correct_predictions = (predicted == targets).sum().item()
    total_predictions = targets.numel()
    return (correct_predictions / total_predictions) * 100.0


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    scheduler=None,
    scaler: Optional[GradScaler] = None
) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    Returns average loss and accuracy.
    """
    model.train()
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    total_loss = 0.0
    total_acc = 0.0
    for batch_idx, (inputs, targets, _) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if scaler and is_distributed:
            loss_val = run_train_step_distributed(model, inputs, targets, optimizer, scheduler, scaler)
            if is_distributed:
                loss_val = aggregate_loss(loss_val, device)
        else:
            loss_val = run_train_step(model, inputs, targets, optimizer, scheduler, device)

        total_loss += loss_val
        with torch.no_grad():
            outputs = model(inputs)
            total_acc += compute_accuracy(outputs, targets)

        if rank == 0 and batch_idx % PRINT_FREQ == 0 and batch_idx != 0:
            avg_loss_so_far = total_loss / (batch_idx + 1)
            current_lr = scheduler.get_last_lr()[0] if scheduler else DEFAULT_LR
            print(
                f"Progress: {batch_idx}/{len(train_loader)} | "
                f"Train Loss: {avg_loss_so_far:.4f} | "
                f"LR: {current_lr:.8f}"
            )

    average_loss = total_loss / len(train_loader)
    average_acc = total_acc / len(train_loader)
    return average_loss, average_acc


def validate_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    val_loader: DataLoader,
    n_classes: int = 1
) -> Tuple[float, Tensor]:
    """
    Validate the model for one epoch.
    Returns average loss and accuracy (per class).
    """
    model.eval()
    is_distributed = dist.is_initialized()

    total_loss = 0.0
    correct_predictions = torch.zeros(n_classes, device=device)
    total_predictions = torch.zeros(n_classes, device=device)

    with torch.no_grad():
        for inputs, targets, weights in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            weights = weights.to(device)

            outputs = model(inputs)
            loss = F.binary_cross_entropy_with_logits(outputs, targets, weight=weights)

            loss_val = loss.item()
            if is_distributed:
                loss_val = aggregate_loss(loss_val, device)

            total_loss += loss_val

            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_mask = (predicted == targets).float()

            for i in range(n_classes):
                valid_targets = (targets[:, i] != -1)
                if valid_targets.any():
                    correct_predictions[i] += ((predicted[:, i] == targets[:, i]) & valid_targets).float().sum()
                    total_predictions[i] += valid_targets.sum()

    average_loss = total_loss / len(val_loader)
    accuracy = (correct_predictions / total_predictions) * 100.0
    return average_loss, accuracy


def main(
    output_dir: str,
    data_dir: str,
    hyperparams: HyperParams,
    train_file: str,
    valid_file: str,
    local_rank: int,
) -> None:
    """
    Main function to train and validate the model.
    """
    set_seeds(SEED_VALUE)
    suppress_warnings()

    run_mode = determine_run_mode()
    device = get_device(run_mode, local_rank)

    model = build_model(device, data_dir)

    if run_mode == RunMode.DISTRIBUTED:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(device)
        model = DDP(model)
    else:
        model.to(device)

    if torch.cuda.is_available():
        model = torch.compile(model)

    num_workers = 4 if torch.cuda.device_count() >= 1 else 0
    train_loader, valid_loader = get_data_loaders(
        data_dir=data_dir,
        train_file=train_file,
        valid_file=valid_file,
        batch_size=hyperparams.batch_size,
        num_workers=num_workers,
    )

    criterion = nn.BCEWithLogitsLoss()
    param_groups = get_params_without_weight_decay_ln(model.named_parameters(), DEFAULT_WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=hyperparams.learning_rate,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        betas=DEFAULT_BETAS,
        eps=DEFAULT_EPS,
    )

    total_steps = len(train_loader) * hyperparams.num_epochs
    warmup_steps = WARMUP_STEPS_GPU if torch.cuda.is_available() else WARMUP_STEPS_CPU
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    early_stopping = EarlyStopping(
        patience=hyperparams.early_stopping_patience,
        verbose=True,
        save_path=os.path.join(output_dir, "best_model.pth")
    )

    rank = dist.get_rank() if dist.is_initialized() else 0

    for epoch in range(hyperparams.num_epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train_loader=train_loader,
            scheduler=scheduler,
        )

        if CHECK_LOSS_NAN and torch.isnan(torch.tensor(train_loss)) and run_mode == RunMode.DISTRIBUTED:
            dist.destroy_process_group()
            break

        val_loss, val_acc = validate_one_epoch(
            model=model,
            criterion=criterion,
            device=device,
            val_loader=valid_loader,
            n_classes=1,
        )

        if rank == 0:
            print(
                f"Epoch: {epoch + 1}/{hyperparams.num_epochs} "
                f"| Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} "
                f"| Val loss: {val_loss:.4f} | Val acc: {val_acc} "
                f"| LR: {scheduler.get_last_lr()[0]:.4f}"
            )

            if early_stopping(val_loss, model):
                if run_mode == RunMode.DISTRIBUTED:
                    dist.destroy_process_group()
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepSeq model on SageMaker.")
    parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--num-epochs", type=int, default=HyperParams.num_epochs)
    parser.add_argument("--batch-size", type=int, default=HyperParams.batch_size)
    parser.add_argument("--learning-rate", type=float, default=HyperParams.learning_rate)
    parser.add_argument("--early-stopping-patience", type=int, default=HyperParams.early_stopping_patience)
    parser.add_argument("--focal-loss-alpha", type=float, default=HyperParams.focal_loss_alpha)
    parser.add_argument("--focal-loss-gamma", type=float, default=HyperParams.focal_loss_gamma)
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument("--train-file", type=str, default="training_combined.csv")
    parser.add_argument("--valid-file", type=str, default="validation_combined.csv")
    parser.add_argument("--transform-type", type=str, default="NONE")
    parser.add_argument("--filter-type", type=str, default="NONE")

    args = parser.parse_args()

    # Validate critical inputs early
    if not args.output_dir:
        raise ValueError("Output directory is required.")
    if not args.data_dir:
        raise ValueError("Data directory is required.")
    if not os.path.isdir(args.data_dir):
        raise ValueError(f"Data directory {args.data_dir} does not exist.")
    
    # convert string to enum
    transform_type = TransformType.from_str(args.transform_type)
    filter_type = FilterType.from_str(args.filter_type)

    hyperparams = HyperParams(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        focal_loss_alpha=args.focal_loss_alpha,
        focal_loss_gamma=args.focal_loss_gamma,
        transform_type=args.transform_type,
        filter_type=args.filter_type,
    )

    main(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        hyperparams=hyperparams,
        train_file=args.train_file,
        valid_file=args.valid_file,
        local_rank=args.local_rank,
    )
