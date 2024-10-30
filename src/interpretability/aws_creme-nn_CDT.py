import boto3
import os
import argparse
import json
from sagemaker import Session
from sagemaker.pytorch import PyTorchModel
import time

import torch
from creme import creme
from creme import utils
import numpy as np
import pandas as pd


def make_a_prediction(self, x, cell_line_name_with_timestamp, args):
    # check to make sure shape is correct
    if len(x.shape) == 2:
        x = x[np.newaxis]
    
    if torch.is_tensor(x):
        x_torch = x.float().to(self.device)
    else:
        x_torch = torch.from_numpy(x).float().to(self.device)

    preds = self.transform(
        data=x_torch,
        data_type="S3Prefix",
        content_type=args.content_type,
        split_type=args.split_type,
        job_name=f"{cell_line_name_with_timestamp}"
    )
    preds = torch.sigmoid(preds).detach().to('cpu').numpy()

    if self.bin_index:
        preds = preds[:, self.bin_index, :]
    if self.track_index:
        preds = preds[..., self.track_index]

    return preds


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
    # Initialize a SageMaker session
    sagemaker_session = Session()
    s3 = boto3.client('s3')
    
    # Create PyTorchModel and run transform jobs
    for cell_line_name, model_artifact_s3_location in args.model_artifact_s3_locations.items():
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
        cell_line_name_with_timestamp = f"{cell_line_name}-{timestamp}"

        # Delete existing files from the specified S3 locations with timestamped paths
        delete_s3_objects(
            s3, 
            bucket_name=args.s3_bucket, 
            prefix=f"{args.input_prefix}/{cell_line_name_with_timestamp}",
        )
        delete_s3_objects(
            s3, 
            bucket_name=args.s3_bucket, 
            prefix=f"{args.output_prefix}/{cell_line_name_with_timestamp}",
        )

    data_dir = os.path.join(root_dir, "data")

    json_path = os.path.join(data_dir, f"jsonl/dataset_{json_id}.jsonl.gz")

    pytorch_model = PyTorchModel(
        model_data=model_artifact_s3_location,
        role=args.iam_role,
        framework_version=args.framework_version,
        py_version=args.py_version,
        source_dir=os.path.join(args.project_path, args.source_dir),
        entry_point=args.entry_point,
        sagemaker_session=sagemaker_session,
        name=f"{args.model_name_prefix}-{cell_line_name_with_timestamp}",
        env={"TS_MAX_RESPONSE_SIZE": "100000000",
            "TS_DEFAULT_STARTUP_TIMEOUT": "600",
            'TS_DEFAULT_RESPONSE_TIMEOUT': '1000',
            "SAGEMAKER_MODEL_SERVER_WORKERS": "4"}
    )

    # Create transformer from PyTorchModel object
    output_path = f"s3://{args.s3_bucket}/{args.output_prefix}/{cell_line_name_with_timestamp}"
    transformer = pytorch_model.transformer(
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        output_path=output_path,
        strategy=args.strategy,
        max_concurrent_transforms=args.max_concurrent_transforms,
        max_payload=args.max_payload,
    )
    setattr(transformer, "predict", make_a_prediction)

    # Start the transform job
    transformer.transform(
        data=inputs,
        data_type="S3Prefix",
        content_type=args.content_type,
        split_type=args.split_type,
        wait=False,
        job_name=f"{cell_line_name_with_timestamp}"
    )

    print(f"Transformation output saved to: {output_path}")

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
    for sample_index in range(data.shape[0]):
        sample_data = data[sample_index].to('cpu').numpy()
        result = creme.context_dependence_test(
            pytorch_model, 
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
