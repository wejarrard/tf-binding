# pretrain.py
import argparse
import os
import warnings
from dataclasses import dataclass
from multiprocessing import cpu_count

import pysam
import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
from multi_tf_dataloader import TFIntervalDataset
from deepseq import DeepSeq
from earlystopping import EarlyStopping
from einops.layers.torch import Rearrange
from enformer_pytorch import Enformer
from finetune import HeadAdapterWrapper
from loss import FocalLoss
from torch.cpu import is_available
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from training_utils import (
    count_directories,
    train_one_epoch,
    transfer_enformer_weights_to_,
    validate_one_epoch,
)
from transformers import get_linear_schedule_with_warmup

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


    num_tfs = 2
    
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

    ############ MODEL ############

    num_cell_lines = 33

    model = DeepSeq.from_hparams(
        dim=1536,
        depth=11,
        heads=8,
        target_length=-1,
        num_cell_lines=num_cell_lines,
        return_augs=True,
        num_downsamples=5,
    ).to(device)

    # model = transfer_enformer_weights_to_(model, transformer_only=True)
    state_dict = torch.load(
        os.path.join(data_dir, "pretrained_weights.pth"), map_location=device
    )
    modified_state_dict = {
        key.replace("_orig_mod.module.", ""): value for key, value in state_dict.items()
    }
    model.load_state_dict(modified_state_dict)

    model.out = nn.Sequential(
        # PrintShape(name="Head In"),
        nn.Linear(model.dim * 2, 1),
        # PrintShape(name="Linear"),
        Rearrange("... () -> ..."),
        nn.Linear(512, num_tfs),
    )

    for param in model.parameters():
        param.requires_grad = True

    # for name, param in model.named_parameters():
    #     if "transformer" in name:
    #         param.requires_grad = False
    # print("Transformer weights frozen")

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

    if torch.cuda.device_count() >= 1:
        num_workers = 6
    else:
        num_workers = 0

    train_dataset = TFIntervalDataset(
        bed_file=os.path.join(data_dir, "FOXA1_AR_train"),
        fasta_file=os.path.join(data_dir, "genome.fa"),
        cell_lines_dir=os.path.join(data_dir, "cell_lines/"),
        num_tfs=num_tfs,
        return_augs=False,
        rc_aug=True,
        shift_augs=(-50, 50),
        context_length=16_384,
        mode="train",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_dataset = TFIntervalDataset(
        bed_file=os.path.join(data_dir, "FOXA1_AR_val"),
        fasta_file=os.path.join(data_dir, "genome.fa"),
        cell_lines_dir=os.path.join(data_dir, "cell_lines/"),
        num_tfs=num_tfs,
        return_augs=False,
        rc_aug=False,
        context_length=16_384,
        mode="train",
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=hyperparams.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
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
    warmup_steps = 1000 if torch.cuda.is_available() else 0

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        verbose=True,
        save_path=f"/opt/ml/model/pretrained_weight.pth",
    )

    ############ TENSORBOARD ############

    writer = SummaryWriter(
        log_dir="/opt/ml/output/tensorboard" if torch.cuda.is_available() else "output"
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
        )
        if rank == 0:
            print(
                f"Epoch: {epoch + 1}/{hyperparams.num_epochs} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.4f}"
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
