# pretrain.py
import argparse
import os
import warnings
from dataclasses import dataclass
from multiprocessing import cpu_count

from enformer_pytorch import Enformer
import pysam
import torch
import torch._dynamo
from torch.cpu import is_available
import torch.distributed as dist
import torch.nn as nn
from earlystopping import EarlyStopping
from einops.layers.torch import Rearrange
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from training_utils import (
    transfer_enformer_weights_to_,
    train_one_epoch,
    validate_one_epoch,
)
from transformers import get_linear_schedule_with_warmup

from data import GenomeIntervalDataset, CELL_LINES
from deepseq import DeepSeq

seed_value = 42
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)
    
torch.autograd.set_detect_anomaly(True)

# hide user warning
warnings.filterwarnings("ignore", category=UserWarning)
pysam.set_verbosity(0)

sm_hosts_str = os.environ.get("SM_HOSTS", "")
sm_hosts = sm_hosts_str.split(",")

if len(sm_hosts) > 1:
    DISTRIBUTED = True
else:
    # Use the number of GPUs as a fallback
    DISTRIBUTED = torch.cuda.device_count() > 1


############ HYPERPARAMETERS ############
@dataclass
class HyperParams:
    num_epochs: int = 50
    batch_size: int = 8 if torch.cuda.is_available() else 1

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


def main(output_dir: str, data_dir: str, hyperparams: HyperParams) -> None:
    ############ DEVICE ############

    # Check for CUDA availability
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        gpu_ok = False
    else:
        if DISTRIBUTED:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(args.local_rank)
            device = torch.device(f"cuda:{args.local_rank}")
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

    if DISTRIBUTED:
        # https://github.com/dougsouza/pytorch-sync-batchnorm-example
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(device)
        model = DDP(model)

    else:
        model.to(device)

    # model = torch.compile(model) if gpu_ok else model
    model = torch.compile(model) if torch.cuda.is_available() else model

    ############ DATA ############

    dataset = GenomeIntervalDataset(
        bed_file=os.path.join(data_dir, "combined.bed"),
        fasta_file=os.path.join(data_dir, "genome.fa"),
        cell_lines_dir=os.path.join(data_dir, "cell_lines/"),
        # cell_lines_dir=os.path.join("/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines"),
        return_augs=False,
        rc_aug=True,
        shift_augs=(-50, 50),
        context_length=16_384,
    )

    if torch.cuda.is_available():
        total_size = len(dataset)
        valid_size = 20_000
        train_size = total_size - valid_size

        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    else:
        # Create a subset of 100 samples from the dataset
        subset_size = 2
        subset_indices = torch.randperm(len(dataset))[:subset_size].tolist()
        subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
        # Divide the subset into training and validation
        valid_size = 1  # Set your validation size
        train_size = subset_size - valid_size
        train_dataset, valid_dataset = random_split(
            subset_dataset, [train_size, valid_size]
        )

    assert (
        train_size > 0
    ), f"The dataset only contains {total_size} samples, but {valid_size} samples are required for the validation set."

    if torch.cuda.device_count() >= 1:
        num_workers = 6
    else:
        num_workers = 0

    print(f"Using {num_workers} workers")

    if DISTRIBUTED:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=args.local_rank,
            drop_last=True    
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=hyperparams.batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=True
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=hyperparams.batch_size,
            sampler=valid_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=hyperparams.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=hyperparams.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
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

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    for epoch in range(hyperparams.num_epochs):
        if DISTRIBUTED:
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)
            

        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            # scaler=scaler,
            scheduler=scheduler,
            train_loader=train_loader,
        )
        if rank == 0:
            if torch.isnan(torch.tensor(train_loss)):
                if DISTRIBUTED:
                    torch.distributed.destroy_process_group()
                break

        val_loss, val_acc = validate_one_epoch(
            model=model,
            val_loader=valid_loader,
            criterion=criterion,
            device=device,
            n_classes=len(CELL_LINES),
            enable_umap=True,
            umap_output_dir=os.path.join(output_dir, "umap_results"),
        )
        if rank == 0:
            print(
                f"Epoch: {epoch + 1}/{hyperparams.num_epochs} | Train loss: {float(train_loss):.4f} | Train acc: {float(train_acc):.4f} | Val loss: {float(val_loss):.4f} | Val acc: {float(val_acc):.4f} | LR: {scheduler.get_last_lr()[0]:.4f}"
            )
            if early_stopping(val_loss, model):
                if DISTRIBUTED:
                    torch.distributed.destroy_process_group()
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
    parser.add_argument(
        "--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0))
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
