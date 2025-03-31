# pretrain.py
import argparse
import os
import warnings
from dataclasses import dataclass

import pysam
import torch
import torch._dynamo
import torch.nn as nn
from earlystopping import EarlyStopping
from torch.utils.data import DataLoader, random_split
from training_utils import (
    transfer_enformer_weights_to_,
    train_one_epoch,
    validate_one_epoch,
)
from transformers import get_linear_schedule_with_warmup
from data import GenomeIntervalDataset, CELL_LINES
from deepseq import DeepSeq
from torch.utils.data import default_collate

seed_value = 42
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)
    
torch.autograd.set_detect_anomaly(True)

# hide user warning
warnings.filterwarnings("ignore", category=UserWarning)
pysam.set_verbosity(0)

# Set distributed flag to False to disable multiprocessing
DISTRIBUTED = False


############ HYPERPARAMETERS ############
@dataclass
class HyperParams:
    num_epochs: int = 50
    batch_size: int = 8 if torch.cuda.is_available() else 4

    learning_rate: float = 5e-4
    early_stopping_patience: int = 2
    focal_loss_alpha: float = 1
    focal_loss_gamma: float = 2


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

def skip_inconsistent_dims_collate(batch):
    """
    Custom collate function that skips batches with inconsistent dimensions.
    
    If tensors in the batch have inconsistent dimensions, return None
    which will be handled in the training loop.
    """
    try:
        # Try the default collation
        return default_collate(batch)
    except RuntimeError as e:
        # Check if the error is due to inconsistent tensor sizes
        if "stack expects each tensor to be equal size" in str(e):
            print(f"Skipping batch with inconsistent dimensions: {str(e)}")
            # Return None to indicate this batch should be skipped
            return None
        else:
            # Re-raise other runtime errors
            raise e


def main(output_dir: str, data_dir: str, hyperparams: HyperParams) -> None:
    ############ DEVICE ############

    # Check for CUDA availability
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        gpu_ok = False
    else:
        device = torch.device("cuda")

        # Checking GPU compatibility
        gpu_ok = torch.cuda.get_device_capability() in (
            (7, 0),
            (8, 0),
            (9, 0),
        )

        if not gpu_ok:
            print(
                "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower than expected."
            )

    ############ MODEL ############

    num_cell_lines = len(CELL_LINES)

    model = DeepSeq.from_hparams(
        dim=1536,
        depth=11,
        heads=8,
        target_length=-1,
        num_cell_lines=num_cell_lines,
        return_augs=True,
        num_downsamples=5,
    ).to(device)

    model = transfer_enformer_weights_to_(model, transformer_only=True)
    # state_dict = torch.load(os.path.join(data_dir,'best_model.pth'), map_location=device)
    # modified_state_dict = {key.replace('_orig_mod.module.', ''): value for key, value in state_dict.items()}
    # model.load_state_dict(modified_state_dict)

    for param in model.parameters():
        param.requires_grad = True

    for name, param in model.named_parameters():
        if "transformer" in name:
            param.requires_grad = False
    print("Transformer weights frozen")

    model.to(device)

    # model = torch.compile(model) if gpu_ok else model
    # model = torch.compile(model) if torch.cuda.is_available() else model

    ############ DATA ############

    # Create separate datasets for training and validation using different bed files
    train_dataset = GenomeIntervalDataset(
        # bed_file=os.path.join(data_dir, "train.bed"),
        bed_file=os.path.join(data_dir, "valid.bed"),
        fasta_file=os.path.join(data_dir, "genome.fa"),
        cell_lines_dir=os.path.join("/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines"),
        # cell_lines_dir=os.path.join(data_dir, "cell_lines/"),
        return_augs=False,
        shift_augs=(0, 0),
        context_length=16_384,
    )
    
    valid_dataset = GenomeIntervalDataset(
        bed_file=os.path.join(data_dir, "valid.bed"),
        fasta_file=os.path.join(data_dir, "genome.fa"),
        cell_lines_dir=os.path.join("/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines"),
        # cell_lines_dir=os.path.join(data_dir, "cell_lines/"),
        return_augs=False,
        shift_augs=(0, 0),
        context_length=16_384,
    )
    
    # For CPU testing with smaller datasets
    if not torch.cuda.is_available():
        # Create small subsets for CPU testing
        train_subset_size = 10
        valid_subset_size = 10
        
        train_subset_indices = torch.randperm(len(train_dataset))[:train_subset_size].tolist()
        valid_subset_indices = torch.randperm(len(valid_dataset))[:valid_subset_size].tolist()
        
        train_dataset = torch.utils.data.Subset(train_dataset, train_subset_indices)
        valid_dataset = torch.utils.data.Subset(valid_dataset, train_subset_indices)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")

    if torch.cuda.device_count() >= 1:
        num_workers = 6
    else:
        num_workers = 0

    print(f"Using {num_workers} workers")

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=skip_inconsistent_dims_collate  # Add this line
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=hyperparams.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=skip_inconsistent_dims_collate  # Add this line
    )

    ############ TRAINING PARAMS ############

    param_groups = get_params_without_weight_decay_ln(
        model.named_parameters(), weight_decay=0.1
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=hyperparams.learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    # scaler = GradScaler()

    total_steps = len(train_loader) * hyperparams.num_epochs
    warmup_steps = 50_000 if torch.cuda.is_available() else 0

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience, verbose=True, save_path=f"/opt/ml/model/best_model.pth"
    )

    ############ TRAINING ############

    for epoch in range(hyperparams.num_epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            # scaler=scaler,
            scheduler=scheduler,
            train_loader=train_loader,
        )
        
        if torch.isnan(torch.tensor(train_loss)):
            break

        val_loss, val_acc = validate_one_epoch(
            model=model,
            val_loader=valid_loader,
            criterion=criterion,
            device=device,
            n_classes=len(CELL_LINES),
            enable_umap=True,
            umap_output_dir=os.path.join(output_dir, "umap_results"),
            umap_samples=10_000,
        )
        
        print(
            f"Epoch: {epoch + 1}/{hyperparams.num_epochs} | Train loss: {float(train_loss):.4f} | Train acc: {float(train_acc):.4f} | Val loss: {float(val_loss):.4f} | Val acc: {float(val_acc):.4f} | LR: {scheduler.get_last_lr()[0]:.4f}"
        )
        
        if early_stopping(val_loss, model):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepSeq model on SageMaker.")
    parser.add_argument(
        "--output-dir", type=str, default=os.environ.get("SM_MODEL_DIR")
    )
    parser.add_argument(
        "--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING")
    )

    # Define command line arguments for hyperparameters with default values directly taken from HyperParams class
    parser.add_argument("--num-epochs", type=int, default=HyperParams.num_epochs)
    parser.add_argument("--batch-size", type=int, default=HyperParams.batch_size)
    parser.add_argument(
        "--learning-rate", type=float, default=HyperParams.learning_rate
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=HyperParams.early_stopping_patience,
    )
    parser.add_argument(
        "--focal-loss-alpha", type=float, default=HyperParams.focal_loss_alpha
    )
    parser.add_argument(
        "--focal-loss-gamma", type=float, default=HyperParams.focal_loss_gamma
    )

    args = parser.parse_args()

    # Create hyperparams instance with values from command line arguments
    hyperparams = HyperParams(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        focal_loss_alpha=args.focal_loss_alpha,
        focal_loss_gamma=args.focal_loss_gamma,
    )

    main(args.output_dir, args.data_dir, hyperparams)