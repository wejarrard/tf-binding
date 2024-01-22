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
    print(f"Saving the Checkpoint: {checkpointing_path}")
    torch.save(
        {
            "epoch": epoch,
            current_batch: current_batch,
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
    print("Resuming training from epoch: {}".format(epoch_number + 1))
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
