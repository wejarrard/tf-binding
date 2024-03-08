# pretrain.py
import argparse
import dataclasses
import os
from dataclasses import dataclass

import lightning as pl
import pysam
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from dataloaders.tf import TFIntervalDataset
from einops.layers.torch import Rearrange
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from models.config import EnformerConfig
from models.deepseq import DeepSeqBase
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy
from utils.scheduler import get_linear_schedule_with_warmup
from utils.training import count_directories, get_params_without_weight_decay_ln

pysam.set_verbosity(0)


def seed_everything(seed: int = 42) -> None:
    # Set seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


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

    checkpoint_path = (
        "/opt/ml/checkpoints"
        if os.path.exists("/opt/ml/checkpoints")
        else "./checkpoints"
    )
    model_output_path = (
        "/opt/ml/model" if os.path.exists("/opt/ml/model") else "./output"
    )
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


############ MODEL ############
class DeepSeq(pl.LightningModule):
    def __init__(
        self, config, train_loader_length: int, number_of_classes=1, state_dict=None
    ):
        super().__init__()
        self.train_acc = BinaryAccuracy(threshold=0.5)
        self.valid_acc = BinaryAccuracy(threshold=0.5)
        self.train_loader_length = train_loader_length

        self.model = DeepSeqBase(config)

        if state_dict:
            state_dict = torch.load(state_dict)
            modified_state_dict = {
                key.replace("_orig_mod.module.", ""): value
                for key, value in state_dict.items()
            }
            self.model.load_state_dict(modified_state_dict)

        self.model.out = nn.Sequential(
            nn.Linear(self.model.dim * 2, 1),
            Rearrange("... () -> ..."),
            nn.Linear(512, number_of_classes),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets, score = batch[0], batch[1], batch[2]
        outputs = self.model(inputs)
        train_loss = F.binary_cross_entropy_with_logits(outputs, targets, weight=score)
        self.train_acc(outputs, targets)
        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_accuracy",
            self.train_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch[0], batch[1]
        outputs = self.model(inputs)
        val_loss = F.binary_cross_entropy_with_logits(outputs, targets)
        self.valid_acc(outputs, targets)
        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_accuracy",
            self.valid_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return val_loss

    def configure_optimizers(self):
        param_groups = get_params_without_weight_decay_ln(
            self.model.named_parameters(), weight_decay=0.1
        )

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=hyperparams.learning_rate,
            weight_decay=0.1,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        total_steps = self.train_loader_length * hyperparams.num_epochs
        warmup_steps = 0.1 * total_steps if torch.cuda.is_available() else 0

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        return [optimizer], [scheduler]


def main(hyperparams: HyperParams) -> None:
    ########### SEED ############
    seed_everything()

    ############ DATALOADERS ############

    num_workers = 6 if torch.cuda.is_available() else 4

    train_dataset = TFIntervalDataset(
        bed_file=os.path.join(hyperparams.data_dir, "AR_ATAC_broadPeak_train"),
        fasta_file=os.path.join(hyperparams.data_dir, "genome.fa"),
        cell_lines_dir=os.path.join(hyperparams.data_dir, "cell_lines/"),
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
        bed_file=os.path.join(hyperparams.data_dir, "AR_ATAC_broadPeak_val"),
        fasta_file=os.path.join(hyperparams.data_dir, "genome.fa"),
        cell_lines_dir=os.path.join(hyperparams.data_dir, "cell_lines/"),
        return_augs=False,
        rc_aug=False,
        context_length=16_384,
        mode="valid",
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=hyperparams.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    ############ MODEL ############

    num_cell_lines = count_directories(
        os.path.join(hyperparams.data_dir, "cell_lines/")
    )

    config = EnformerConfig(
        dim=1536,
        depth=11,
        heads=8,
        target_length=-1,
        num_cell_lines=num_cell_lines,
        num_downsamples=5,
    )

    model = DeepSeq(config, train_loader_length=len(train_loader))

    for param in model.parameters():
        param.requires_grad = True

    ############ TRAINING ############

    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=hyperparams.early_stopping_patience,
        verbose=True,
        mode="min",
    )

    # Checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=hyperparams.checkpoint_path,
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        save_weights_only=False,
        filename=f"best_model",
        verbose=True,
        mode="min",
        every_n_train_steps=1,
        enable_version_counter=False,
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=hyperparams.num_epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        gradient_clip_val=None,
        deterministic=True,
        log_every_n_steps=1,
    )

    if os.path.exists(f"{hyperparams.checkpoint_path}/best_model.ckpt"):
        print("\n \n \n Resuming Training \n \n \n")
        trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=f"{hyperparams.checkpoint_path}/best_model.ckpt"
        )       
    else:
        print("\n \n \n Starting new training run \n \n \n")
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )


if __name__ == "__main__":
    hyperparams = HyperParams()
    hyperparams.parse_arguments("Train DeepSeq model on SageMaker.")

    main(hyperparams)
