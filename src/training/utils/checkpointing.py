import argparse
from dataclasses import dataclass
import dataclasses
import os
import shutil

import numpy as np
import torch



def save_checkpoint(
    model,
    optimizer,
    scheduler,
    early_stopping,
    epoch,
    hyperparams,
    total_loss=None,
    correct_predictions=None,
    total_predictions=None,
    current_batch=0,
    save_best_model=False,
):
    checkpointing_path = hyperparams.checkpoint_path + "/checkpoint.pth"
    if os.path.exists(checkpointing_path):
        os.remove(checkpointing_path)

    print(f"Saving the Checkpoint: {checkpointing_path}")
    torch.save(
        {
            "epoch": epoch,
            "current_batch": current_batch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "early_stopping_state": early_stopping.state_dict(),
            "total_loss": total_loss,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
        },
        checkpointing_path,
    )

    if save_best_model:
        source_best_model_path = f"{hyperparams.model_output_path}/best_model.pth"
        target_best_model_path = f"{hyperparams.checkpoint_path}/best_model.pth"
        shutil.copyfile(source_best_model_path, target_best_model_path)
        print(f"Best model saved to {target_best_model_path}")


def load_checkpoint(model, optimizer, scheduler, early_stopping, hyperparams):
    print("--------------------------------------------")
    checkpoint_path = hyperparams.checkpoint_path + "/checkpoint.pth"
    print(f"Loading Checkpoint From: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    early_stopping.load_state_dict(checkpoint["early_stopping_state"])

    current_batch = checkpoint.get("current_batch", 0)
    epoch_number = checkpoint["epoch"]
    total_loss = checkpoint.get("total_loss", None)
    correct_predictions = checkpoint.get("correct_predictions", None)
    total_predictions = checkpoint.get("total_predictions", None) 

    if os.path.exists(f"{hyperparams.checkpoint_path}/best_model.pth"):
        source_best_model_path = f"{hyperparams.checkpoint_path}/best_model.pth"
        target_best_model_path = f"{hyperparams.model_output_path}/best_model.pth"
        shutil.copyfile(source_best_model_path, target_best_model_path)
        print(f"Best model saved to {target_best_model_path}")

    print(f"Checkpoint File Loaded - Epoch Number: {epoch_number} - Loss: {total_loss}")
    print("Resuming training from epoch: {}".format(epoch_number))
    print("--------------------------------------------")

    return (
        model,
        optimizer,
        scheduler,
        epoch_number,
        early_stopping,
        current_batch,
        total_loss,
        correct_predictions,
        total_predictions,
    )


if __name__ == "__main__":
    import torch.nn as nn
    from earlystopping import EarlyStopping


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

    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)
    # Create dummy model, optimizer, scheduler, and hyperparams
    model = DummyModel()
    hyperparams = HyperParams()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    early_stopping = EarlyStopping(
        patience=hyperparams.early_stopping_patience,
        verbose=True,
        save_path=f"{hyperparams.model_output_path}/best_model.pth",
    )
    

    # Ensure the checkpoint and output directories exist
    os.makedirs(hyperparams.checkpoint_path, exist_ok=True)
    os.makedirs(hyperparams.model_output_path, exist_ok=True)

    # Save a checkpoint
    save_checkpoint(model, optimizer, scheduler, early_stopping, epoch=1, hyperparams=hyperparams)
    save_checkpoint(model, optimizer, scheduler, early_stopping, epoch=1, hyperparams=hyperparams)


    # Load the checkpoint
    loaded_components = load_checkpoint(model, optimizer, scheduler, early_stopping, hyperparams)

    # Check if the loaded components match expected values
    loaded_model, loaded_optimizer, loaded_scheduler, loaded_epoch, loaded_early_stopping, *rest = loaded_components
    print("Model state dict:", loaded_model.state_dict())
    print("Optimizer state dict:", loaded_optimizer.state_dict())
    print("Scheduler state dict:", loaded_scheduler.state_dict())
    print("Loaded epoch:", loaded_epoch)import argparse
from dataclasses import dataclass
import dataclasses
import os
import shutil

import numpy as np
import torch



def save_checkpoint(
    model,
    optimizer,
    scheduler,
    early_stopping,
    epoch,
    hyperparams,
    total_loss=None,
    correct_predictions=None,
    total_predictions=None,
    current_batch=0,
    save_best_model=False,
):
    checkpointing_path = hyperparams.checkpoint_path + "/checkpoint.pth"
    if os.path.exists(checkpointing_path):
        os.remove(checkpointing_path)

    print(f"Saving the Checkpoint: {checkpointing_path}")
    torch.save(
        {
            "epoch": epoch,
            "current_batch": current_batch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "early_stopping_state": early_stopping.state_dict(),
            "total_loss": total_loss,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
        },
        checkpointing_path,
    )

    if save_best_model:
        source_best_model_path = f"{hyperparams.model_output_path}/best_model.pth"
        target_best_model_path = f"{hyperparams.checkpoint_path}/best_model.pth"
        shutil.copyfile(source_best_model_path, target_best_model_path)
        print(f"Best model saved to {target_best_model_path}")


def load_checkpoint(model, optimizer, scheduler, early_stopping, hyperparams):
    print("--------------------------------------------")
    checkpoint_path = hyperparams.checkpoint_path + "/checkpoint.pth"
    print(f"Loading Checkpoint From: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    early_stopping.load_state_dict(checkpoint["early_stopping_state"])

    current_batch = checkpoint.get("current_batch", 0)
    epoch_number = checkpoint["epoch"]
    total_loss = checkpoint.get("total_loss", None)
    correct_predictions = checkpoint.get("correct_predictions", None)
    total_predictions = checkpoint.get("total_predictions", None) 

    if os.path.exists(f"{hyperparams.checkpoint_path}/best_model.pth"):
        source_best_model_path = f"{hyperparams.checkpoint_path}/best_model.pth"
        target_best_model_path = f"{hyperparams.model_output_path}/best_model.pth"
        shutil.copyfile(source_best_model_path, target_best_model_path)
        print(f"Best model saved to {target_best_model_path}")

    print(f"Checkpoint File Loaded - Epoch Number: {epoch_number} - Loss: {total_loss}")
    print("Resuming training from epoch: {}".format(epoch_number))
    print("--------------------------------------------")

    return (
        model,
        optimizer,
        scheduler,
        epoch_number,
        early_stopping,
        current_batch,
        total_loss,
        correct_predictions,
        total_predictions,
    )


if __name__ == "__main__":
    import torch.nn as nn
    from earlystopping import EarlyStopping


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

    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)
    # Create dummy model, optimizer, scheduler, and hyperparams
    model = DummyModel()
    hyperparams = HyperParams()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    early_stopping = EarlyStopping(
        patience=hyperparams.early_stopping_patience,
        verbose=True,
        save_path=f"{hyperparams.model_output_path}/best_model.pth",
    )
    

    # Ensure the checkpoint and output directories exist
    os.makedirs(hyperparams.checkpoint_path, exist_ok=True)
    os.makedirs(hyperparams.model_output_path, exist_ok=True)

    # Save a checkpoint
    save_checkpoint(model, optimizer, scheduler, early_stopping, epoch=1, hyperparams=hyperparams)
    save_checkpoint(model, optimizer, scheduler, early_stopping, epoch=1, hyperparams=hyperparams)


    # Load the checkpoint
    loaded_components = load_checkpoint(model, optimizer, scheduler, early_stopping, hyperparams)

    # Check if the loaded components match expected values
    loaded_model, loaded_optimizer, loaded_scheduler, loaded_epoch, loaded_early_stopping, *rest = loaded_components
    print("Model state dict:", loaded_model.state_dict())
    print("Optimizer state dict:", loaded_optimizer.state_dict())
    print("Scheduler state dict:", loaded_scheduler.state_dict())
    print("Loaded epoch:", loaded_epoch)