import io
import sys
import os
import logging
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from einops.layers.torch import Rearrange

from creme import creme
# from creme import utils
import numpy as np
import pandas as pd


root_dir = "/Users/wjohns/work/quigley_lab_local/projects/tf-binding/repo/src/inference"

if root_dir not in sys.path:
    sys.path.append(root_dir)

from scripts.deepseq import DeepSeq
from scripts.dataloader import JSONLinesDataset


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return torch.device(device)


def load_model(model_dir: str, device: str, num_tfs=1):

    model = DeepSeq.from_hparams(
        dim=1536,
        depth=11,
        heads=8,
        target_length=-1,
        num_cell_lines=num_tfs,
        return_augs=True,
        num_downsamples=3,
    )

    model.out = nn.Sequential(
        nn.Linear(model.dim * 2, num_tfs),
        Rearrange("... c o -> ... o c"),
        nn.Linear(512, 1),
        nn.Flatten(),
    ).to(device)


    state_dict = torch.load(
        os.path.join(model_dir, "best_model.pth"), map_location=device
    )

    modified_state_dict = {
        key.replace("_orig_mod.", ""): value for key, value in state_dict.items()
    }
    model.load_state_dict(modified_state_dict)


    model.to(device)

    return model


def make_a_prediction(self, x):
    # check to make sure shape is correct
    if len(x.shape) == 2:
        x = x[np.newaxis]
    
    if torch.is_tensor(x):
        x_torch = x.float().to(self.device)
    else:
        x_torch = torch.from_numpy(x).float().to(self.device)
    preds = self(x_torch)
    preds = torch.sigmoid(preds).detach().to('cpu').numpy()

    if self.bin_index:
        preds = preds[:, self.bin_index, :]
    if self.track_index:
        preds = preds[..., self.track_index]

    return preds


def model_fn(model_dir, device=None):
    if not device:
        device = get_device()
    model = load_model(model_dir, device=device, num_tfs=1)
    logger.info("Model loaded successfully.")
    return model.to(device)


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/jsonlines':
        logger.info("Reading JSONLines dataset")
        file_stream = io.BytesIO(request_body)
        dataset = JSONLinesDataset(file_stream=file_stream, num_tfs=1, compressed=True)
    else:
        raise ValueError(f"Unsupported content type or request body type: {request_content_type}, {type(request_body)}")
    
    return dataset


def shuffle_tensor(tensor, dim_to_shuffle:int=1, device='cpu'):
    indices = torch.randperm(tensor.size(dim_to_shuffle)).to(device)
    shuffled_tensor = tensor.index_select(dim_to_shuffle, indices).to(device)

    return shuffled_tensor


def add_atac(X, atac, device='cpu'):
    reshaped = torch.cat((X, atac.unsqueeze(-1)), dim=-1).to(device)
    
    return reshaped


def process_atac(seq_data, atac_data, dim_to_shuffle:int=1, device='cpu'):
    shuffled_atac = shuffle_tensor(atac_data, dim_to_shuffle=dim_to_shuffle, device=device)
    return add_atac(seq_data, atac=shuffled_atac, device=device)


def main(n_shuffles, shuffle_atac, json_id, output_dir):
    device = get_device()
    model_dir = os.path.join(root_dir, "models/AR-Full")
    data_dir = os.path.join(root_dir, "data")

    n_jsons = len(os.listdir(os.path.join(data_dir, "jsonl")))

    if not json_id:
        json_id = int(torch.randint(low=1, high=n_jsons+1, size=(1, 1)))
        # used_jsons = {1, 74, 73}
        # if json_id in used_jsons:
        #     while json_id in used_jsons:
        #         json_id = int(torch.randint(low=1, high=n_jsons+1, size=(1, 1)))
        print(json_id)

    json_path = os.path.join(data_dir, f"jsonl/dataset_{json_id}.jsonl.gz")

    setattr(DeepSeq, "predict", make_a_prediction)
    model = model_fn(model_dir, device=device)

    model.seq_length = 196608 # Should we change this?
    model.head = 'human'
    model.track_index = None # WTF is this?
    model.bin_index = None # WTF is this?

    with open(json_path, "rb") as f:
        request_body = f.read()
    dataset = input_fn(request_body, "application/jsonlines")

    data = torch.stack([dataset.data[i]['input'].to(device) for i in range(len(dataset.data))])
    seq_data = torch.stack([dataset.data[i]['input'][..., 0:4].to(device) for i in range(len(dataset.data))])
    atac_data = torch.stack([dataset.data[i]['input'][..., 4].to(device) for i in range(len(dataset.data))])
    shuffled_data = process_atac(seq_data, atac_data, dim_to_shuffle=1, device=device)
    tile_data = [[dataset.data[i]['start'], dataset.data[i]['end']] 
                for i in range(len(dataset.data))]
    targets = torch.stack([dataset.data[i]['target'] for i in range(len(dataset.data))]).flatten()

    data.shape, seq_data.shape, atac_data.shape, len(tile_data)


    if shuffle_atac: 
        data = shuffled_data

    
    cdt_results = list()
    for sample_index in tqdm(range(data.shape[0])):
        sample_data = data[sample_index].to('cpu').numpy()
        result = creme.context_dependence_test(
            model, 
            sample_data, 
            tile_pos=tile_data[sample_index],
            num_shuffle=n_shuffles,
            dims_to_exclude=1,
            mean=False,
            )
        cdt_results.append(result)


    wt_preds = [cdt_results[i][0].flatten()[0] for i in range(len(cdt_results))]
    mt_preds = [cdt_results[i][1] for i in range(len(cdt_results))]

    deltas = [mt_preds[i].mean() - wt_preds[i] for i in range(len(wt_preds))]
    # quotients = [mt_preds[i].mean() / wt_preds[i] for i in range(len(wt_preds))]


    mt_means = [i.mean() for i in mt_preds]

    cdt_df = pd.DataFrame({"wild_type": wt_preds,
                           "mutant":    mt_means,
                           "deltas":    deltas,
                           "target":    targets,})

    cdt_df.to_csv(os.path.join(output_dir, f"AR_context_dependence_test_{json_id}.csv"), index=True)

    torch_mt = torch.tensor(mt_preds).squeeze(-1)
    mt_df = pd.DataFrame()
    for i in range(n_shuffles):
        mt_df[f"shuffle_{i+1}"] = torch_mt[..., i].tolist()

    mt_df.to_csv(os.path.join(output_dir, f"AR_CDT_mutant_predictions_{json_id}.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Creme-nn's Contenxt Dependency Test.")

    parser.add_argument(
        '--n_shuffles',
        type=int,
        default=50,
        help="How many dinucleotide shuffles to predict per sample?",
    )
    parser.add_argument(
        "--shuffle_atac",
        type=bool,
        default=False,
        help="shuffle ATAC channel (TRUE) or hold constant (FALSE)?",
    )
    parser.add_argument(
        "--json_id",
        type=int,
        default=None,
        help="If there is a particular JSON file you wish to use, add it here.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/wjohns/work",
        help="Where to save CSV files?",
    )


    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(
        n_shuffles=args.n_shuffles,
        shuffle_atac=args.shuffle_atac,
        json_id=args.json_id,
        output_dir=args.output_dir,
    )
