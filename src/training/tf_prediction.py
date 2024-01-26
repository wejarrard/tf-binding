# pretrain.py
import argparse
import dataclasses
import os
import warnings
from collections import Counter
from dataclasses import dataclass

import pysam
import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
from dataloaders.tf import TFIntervalDataset
from einops.layers.torch import Rearrange
from models.deepseq import DeepSeq
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from utils.processing import get_weights
from utils.checkpointing import load_checkpoint, save_checkpoint
from utils.earlystopping import EarlyStopping
from utils.loss import FocalLoss
from utils.training import (
    count_directories,
    get_params_without_weight_decay_ln,
    train_one_epoch,
    validate_one_epoch,
)

torch.autograd.anomaly_mode.set_detect_anomaly(True)

seed_value = 42
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

# hide user warning
warnings.filterwarnings("ignore", category=UserWarning)
pysam.set_verbosity(0)

sm_hosts_str = os.environ.get("SM_HOSTS", "")
DISTRIBUTED = len(sm_hosts_str.split(",")) > 1 or torch.cuda.device_count() > 1


############ HYPERPARAMETERS ############
@dataclass
class HyperParams:
    # Hyperparameters and environment variable-based parameters
    num_epochs: int = 50
    batch_size: int = 8 if torch.cuda.is_available() else 1
    learning_rate: float = 5e-4
    early_stopping_patience: int = 2
    max_grad_norm: float = 0.2
    log_frequency: int = 1_000 if torch.cuda.is_available() else 1
    # focal_loss_alpha: float = 1
    # focal_loss_gamma: float = 2


    checkpoint_path = "/opt/ml/checkpoints" if os.path.exists("/opt/ml/checkpoints") else "./checkpoints"
    model_output_path = "/opt/ml/model" if os.path.exists("/opt/ml/model") else "./output"
    data_dir: str = os.environ.get("SM_CHANNEL_TRAINING", "./data")
    local_rank: int = int(os.environ.get("LOCAL_RANK", 0))

    def parse_arguments(self, description: str):
        parser = argparse.ArgumentParser(description=description)
        for field in dataclasses.fields(self):
            parser.add_argument(
                f'--{field.name.replace("_", "-")}',
                type=field.type,
                default=getattr(self, field.name),
            )
        args = parser.parse_args()
        for field in dataclasses.fields(self):
            if hasattr(args, field.name):
                setattr(self, field.name, getattr(args, field.name))


def main(hyperparams: HyperParams) -> None:
    ########### DEVICE ############

    # Initialize distributed process group
    if torch.cuda.is_available() and DISTRIBUTED:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(hyperparams.local_rank)

    # Determine the device
    device = torch.device(
        f"cuda:{hyperparams.local_rank}"
        if torch.cuda.is_available() and DISTRIBUTED
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    ############ MODEL ############

    num_cell_lines = count_directories(
        os.path.join(hyperparams.data_dir, "cell_lines/")
    )

    model = DeepSeq.from_hparams(
        dim=1536,
        depth=11,
        heads=8,
        target_length=-1,
        num_cell_lines=num_cell_lines,
        return_augs=True,
        num_downsamples=5,
    )

    state_dict = torch.load(
        os.path.join(hyperparams.data_dir, "pretrained_weights.pth"),
        map_location=device,
    )
    modified_state_dict = {
        key.replace("_orig_mod.module.", ""): value for key, value in state_dict.items()
    }
    model.load_state_dict(modified_state_dict)

    model.out = nn.Sequential(
        nn.Linear(model.dim * 2, 1),
        Rearrange("... () -> ..."),
        nn.Linear(512, 1),
    )

    for param in model.parameters():
        param.requires_grad = True

    if DISTRIBUTED:
        # https://github.com/dougsouza/pytorch-sync-batchnorm-example
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(device)
        model = DDP(model)
    else:
        model = model.to(device)

    # model = torch.compile(model) if gpu_ok else model
    model = torch.compile(model) if torch.cuda.is_available() else model

    ############ DATA ############

    dataset = TFIntervalDataset(
        bed_file=os.path.join(hyperparams.data_dir, "AR_ATAC_broadPeak"),
        fasta_file=os.path.join(hyperparams.data_dir, "genome.fa"),
        cell_lines_dir=os.path.join(hyperparams.data_dir, "cell_lines/"),
        return_augs=False,
        rc_aug=True,
        shift_augs=(-50, 50),
        context_length=16_384,
    )

    weights = get_weights(os.path.join(hyperparams.data_dir, "AR_ATAC_broadPeak"))
    
 
    total_size = len(dataset)
    valid_size = 20_000
    train_size = total_size - valid_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    num_workers = 6 if torch.cuda.device_count() >= 1 else 0

    if hyperparams.local_rank == 0:
        print(f"Using {num_workers} workers")

    if DISTRIBUTED:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=hyperparams.local_rank,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=hyperparams.batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=hyperparams.batch_size,
            sampler=valid_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=hyperparams.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
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


    # Instantiate FocalLoss with the class weights
    criterion = FocalLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=hyperparams.learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    total_steps = len(train_loader) * hyperparams.num_epochs
    warmup_steps = 1000 if torch.cuda.is_available() else 0

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    early_stopping = EarlyStopping(
        patience=hyperparams.early_stopping_patience,
        verbose=True,
        save_path=f"{hyperparams.model_output_path}/best_model.pth",
    )

    # Load 
    if not os.path.isfile(hyperparams.checkpoint_path + "/checkpoint.pth"):
        epoch_number = 0
        current_batch = 0
        total_loss = None
        correct_predictions = None
        total_predictions = None
    else:
        (
            model,
            optimizer,
            scheduler,
            epoch_number,
            early_stopping,
            current_batch,
            total_loss,
            correct_predictions,
            total_predictions,
        ) = load_checkpoint(model, optimizer, scheduler, early_stopping, hyperparams)

    ############ TENSORBOARD ############ TODO: Add in Tensorboard support

    writer = SummaryWriter(
        log_dir="/opt/ml/output/tensorboard" if torch.cuda.is_available() else "output"
    )
    ############ TRAINING ############

    for epoch in range(epoch_number, hyperparams.num_epochs):
        if DISTRIBUTED:
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            train_loader=train_loader,
            max_grad_norm=hyperparams.max_grad_norm,
            log_frequency=hyperparams.log_frequency,
            current_batch=current_batch,
            epoch=epoch,
            hyperparams=hyperparams,
            early_stopping=early_stopping,
            total_loss=total_loss,
            correct_predictions=correct_predictions,
            total_predictions=total_predictions,
        )
        if hyperparams.local_rank == 0:
            if torch.isnan(torch.tensor(train_loss)):
                if DISTRIBUTED:
                    dist.destroy_process_group()
                break

        val_loss, val_acc = validate_one_epoch(
            model=model,
            val_loader=valid_loader,
            criterion=criterion,
            device=device,
        )

        if hyperparams.local_rank == 0:
            print(
                f"Epoch: {epoch + 1}/{hyperparams.num_epochs} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.4f}"
            )

            if early_stopping(val_loss, model):
                if DISTRIBUTED:
                    dist.destroy_process_group()
                break

            total_loss, correct_predictions, total_predictions = None, None, None

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                early_stopping,
                epoch,
                hyperparams,
                save_best_model=True,
            )


if __name__ == "__main__":
    hyperparams = HyperParams()
    hyperparams.parse_arguments("Train DeepSeq model on SageMaker.")

    main(hyperparams)
