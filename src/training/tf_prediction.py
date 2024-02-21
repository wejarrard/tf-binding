# pretrain.py
import argparse
import dataclasses
import os
from dataclasses import dataclass

import lightning as pl
import torch
import torch._dynamo
import torch.nn as nn
from dataloaders.tf import TFIntervalDataset
from einops.layers.torch import Rearrange
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from models.config import EnformerConfig
from models.deepseq import DeepSeqBase
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy
from transformers import get_linear_schedule_with_warmup
from utils.training import count_directories, get_params_without_weight_decay_ln


def seed_everything(seed: int = 42) -> None:
    # Set seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms


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
        else "../misc/data/checkpoints"
    )
    model_output_path = (
        "/opt/ml/model" if os.path.exists("/opt/ml/model") else "../misc/output"
    )
    data_dir: str = os.environ.get("SM_CHANNEL_TRAINING", "../misc/data")
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

        self.train_loader_length = train_loader_length

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets, data_inds = batch[0], batch[1], batch[2]
        outputs = self.model(inputs)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(outputs, targets)
        accuracy = BinaryAccuracy(threshold=0.5)
        accuracy(outputs, targets)
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, data_inds = batch[0], batch[1], batch[2]
        outputs = self.model(inputs)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(outputs, targets)
        accuracy = BinaryAccuracy(threshold=0.5)
        accuracy(outputs, targets)
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)
        return loss

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

    num_workers = 6 if torch.cuda.is_available() else 0

    train_dataset = TFIntervalDataset(
        bed_file=os.path.join(hyperparams.data_dir, "AR_ATAC_broadPeak"),
        fasta_file=os.path.join(hyperparams.data_dir, "genome.fa"),
        cell_lines_dir=os.path.join(hyperparams.data_dir, "cell_lines/"),
        return_augs=False,
        rc_aug=True,
        shift_augs=(-50, 50),
        context_length=16_384,
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
        bed_file=os.path.join(hyperparams.data_dir, "AR_ATAC_broadPeak"),
        fasta_file=os.path.join(hyperparams.data_dir, "genome.fa"),
        cell_lines_dir=os.path.join(hyperparams.data_dir, "cell_lines/"),
        return_augs=False,
        rc_aug=True,
        shift_augs=(-50, 50),
        context_length=16_384,
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

    # weights = get_weights(os.path.join(hyperparams.data_dir, "AR_ATAC_broadPeak"))

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
        every_n_train_steps=500,
        enable_version_counter=False,
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=hyperparams.num_epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        gradient_clip_val=None,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )


if __name__ == "__main__":
    hyperparams = HyperParams()
    hyperparams.parse_arguments("Train DeepSeq model on SageMaker.")

    main(hyperparams)
