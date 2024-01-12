# pretrain.py
# pretrain.py
import argparse
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass

import polars as pl
import pysam
import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
from dataloaders.tf import ValidationGenomeIntervalDataset
from einops.layers.torch import Rearrange
from models.deepseq import DeepSeq
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split

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

def validate_one_epoch(model, criterion: nn.Module, device: torch.device, val_loader: DataLoader):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # Additional list to store data for BED file
    bed_file_data = []

    # Dictionaries to track accuracies and misclassifications
    cell_line_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
    chromosome_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
    misclassifications = {"positive_as_negative": 0, "negative_as_positive": 0}

    is_distributed = torch.distributed.is_initialized()

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            inputs, targets, chr_name, start, end, cell_line, label = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2],
                batch[3],
                batch[4],
                batch[5],
                batch[6],
            )

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if is_distributed:
                loss_tensor = torch.tensor([loss.item()], device=device)
                torch.distributed.all_reduce(loss_tensor)
                loss_val = loss_tensor.item() / torch.distributed.get_world_size()
            else:
                loss_val = loss.item()

            total_loss += loss_val
            
            output_data = outputs.data.cpu().numpy().flatten()
            for i in range(len(chr_name)):
                bed_entry = (
                    chr_name[i],
                    str(start[i]),
                    str(end[i]),
                    cell_line[i],
                    str(label[i]),
                    str(output_data[i])
                )
                bed_file_data.append(bed_entry)

            predicted = (outputs.data > 0.5).float()
            correct = predicted == targets

            # Update general accuracy
            correct_predictions += correct.sum().item()
            total_predictions += targets.numel()

            # Update cell line and chromosome accuracies
            for i in range(len(correct)):
                cell_line_accuracy[cell_line[i]]["correct"] += correct[i].item()
                cell_line_accuracy[cell_line[i]]["total"] += 1
                chromosome_accuracy[chr_name[i]]["correct"] += correct[i].item()
                chromosome_accuracy[chr_name[i]]["total"] += 1

                # Update misclassifications
                if label[i] == 1 and predicted[i] == 0:
                    misclassifications["positive_as_negative"] += 1
                elif label[i] == 0 and predicted[i] == 1:
                    misclassifications["negative_as_positive"] += 1

    average_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions * 100

    
    # Write the data to a BED file
    bed_file_path = "/opt/ml/model/validation_results.bed"
    with open(bed_file_path, "w") as bed_file:
        for entry in bed_file_data:
            bed_file.write("\t".join(entry) + "\n")
    # Calculate accuracies and fractions for each cell line and chromosome
    cell_line_acc_str = {}
    chromosome_acc_str = {}
    for key in cell_line_accuracy:
        fraction = f"{cell_line_accuracy[key]['correct']} / {cell_line_accuracy[key]['total']}"
        percentage = (cell_line_accuracy[key]['correct'] / cell_line_accuracy[key]['total']) * 100
        cell_line_acc_str[key] = {"fraction": fraction, "percentage": percentage}

    for key in chromosome_accuracy:
        fraction = f"{chromosome_accuracy[key]['correct']} / {chromosome_accuracy[key]['total']}"
        percentage = (chromosome_accuracy[key]['correct'] / chromosome_accuracy[key]['total']) * 100
        chromosome_acc_str[key] = {"fraction": fraction, "percentage": percentage}

    return (
        average_loss,
        accuracy,
        cell_line_acc_str,
        chromosome_acc_str,
        misclassifications,
    )



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

    model = DeepSeq.from_hparams(
        dim=1536,
        depth=11,
        heads=8,
        target_length=-1,
        num_cell_lines=1,
        return_augs=True,
        num_downsamples=5,
    ).to(device)

    # model = transfer_enformer_weights_to_(model, transformer_only=True)
    state_dict = torch.load(
        os.path.join(data_dir, "tf_finetuned_weights.pth"), map_location=device
    )
    modified_state_dict = {
        key.replace("_orig_mod.", ""): value for key, value in state_dict.items()
    }
    model.load_state_dict(modified_state_dict)

    for param in model.parameters():
        param.requires_grad = True

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

    dataset = ValidationGenomeIntervalDataset(
        bed_file=os.path.join(data_dir, "AR_ATAC_broadPeak"),
        fasta_file=os.path.join(data_dir, "genome.fa"),
        cell_lines_dir=os.path.join(data_dir, "cell_lines/"),
        return_augs=False,
        rc_aug=False,
        shift_augs=(0, 0),
        context_length=16_384,
    )

#     total_size = len(dataset)
#     valid_size = 20_000
#     train_size = total_size - valid_size

#     train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    if torch.cuda.device_count() >= 1:
        num_workers = 6
    else:
        num_workers = 0

    print(f"Using {num_workers} workers")

    if DISTRIBUTED:
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=True,
        )
        valid_loader = DataLoader(
            dataset,
            batch_size=hyperparams.batch_size,
            sampler=valid_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        valid_loader = DataLoader(
            dataset,
            batch_size=hyperparams.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    ############ Validation ############


    criterion = nn.BCEWithLogitsLoss()

    (
        average_loss,
        accuracy,
        cell_line_accuracy,
        chromosome_accuracy,
        misclassifications,
    ) = validate_one_epoch(
        model=model,
        val_loader=valid_loader,
        criterion=criterion,
        device=device,
    )

    # Assuming the function call you provided has been executed and the results are stored in the respective variables
    print(f"Validation Results:\n")

    print(f"Average Loss: {average_loss:.4f}")
    print(f"Overall Accuracy: {accuracy:.2f}%\n")

    print("Accuracy by Cell Line:")
    for cell_line, acc in cell_line_accuracy.items():
        print(f"  - {cell_line}: {acc}%")

    print("\nAccuracy by Chromosome:")
    for chromosome, acc in chromosome_accuracy.items():
        print(f"  - {chromosome}: {acc}%")

    print("\nMisclassifications:")
    print(
        f"  - Positive labels misclassified as Negative: {misclassifications['positive_as_negative']}"
    )
    print(
        f"  - Negative labels misclassified as Positive: {misclassifications['negative_as_positive']}"
    )




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
