import boto3
import os
import argparse
import json
from sagemaker import Session
from sagemaker.pytorch import PyTorchModel
import time

# Function to delete all objects in a specified S3 bucket/prefix
def delete_s3_objects(s3_client, bucket_name, prefix=""):
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if 'Contents' in response:
        for item in response['Contents']:
            s3_client.delete_object(Bucket=bucket_name, Key=item['Key'])
        print(f"Deleted all objects in {bucket_name}/{prefix}")
    else:
        print(f"No objects found in {bucket_name}/{prefix} to delete.")

# Function to parse the model paths from command-line input
def parse_model_paths(json_input):
    try:
        return json.loads(json_input)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return {}

def main(args):
    # Initialize a SageMaker session
    sagemaker_session = Session()
    s3 = boto3.client('s3')

    # Delete existing files from the specified S3 locations
    delete_s3_objects(s3, bucket_name=args.s3_bucket, prefix=args.input_prefix)
    for key in args.model_artifact_s3_locations:
        delete_s3_objects(s3, bucket_name=args.s3_bucket, prefix=f"{args.output_prefix}/{key}")

    # Upload new files to the specified S3 location
    inputs = sagemaker_session.upload_data(path=args.local_dir, bucket=args.s3_bucket, key_prefix=args.input_prefix)
    print(f"Input spec: {inputs}")

    # Create PyTorchModel and run transform jobs
    for cell_line_name, model_artifact_s3_location in args.model_artifact_s3_locations.items():
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
        cell_line_name_with_timestamp = f"{cell_line_name}-{timestamp}"
        pytorch_model = PyTorchModel(
            model_data=model_artifact_s3_location,
            role=args.iam_role,
            framework_version=args.framework_version,
            py_version=args.py_version,
            source_dir=os.path.join(args.project_path, args.source_dir),
            entry_point=args.entry_point,
            sagemaker_session=sagemaker_session,
            name=f"{args.model_name_prefix}-{cell_line_name_with_timestamp}"
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

if __name__ == "__main__":
    # Set up argument parser to accept configurable parameters
    parser = argparse.ArgumentParser(description='Run SageMaker transform jobs with specified model paths.')

    # Project and directory configurations
    parser.add_argument(
        '--project_path',
        type=str,
        default="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding",
        help='Root path of the project (default: "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding")'
    )
    parser.add_argument(
        '--local_dir',
        type=str,
        default="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/jsonl",
        help='Local directory containing input data (default: "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/jsonl")'
    )
    parser.add_argument(
        '--source_dir',
        type=str,
        default="src/inference/scripts",
        help='Relative path to the source scripts for inference (default: "src/inference/scripts")'
    )
    parser.add_argument(
        '--entry_point',
        type=str,
        default='inference.py',
        help='Entry point script for inference (default: "inference.py")'
    )

    # AWS and SageMaker configurations
    parser.add_argument(
        '--iam_role',
        type=str,
        default="arn:aws:iam::016114370410:role/tf-binding-sites",
        help='IAM role ARN for SageMaker (default: "arn:aws:iam::016114370410:role/tf-binding-sites")'
    )
    parser.add_argument(
        '--s3_bucket',
        type=str,
        default="tf-binding-sites",
        help='S3 bucket name (default: "tf-binding-sites")'
    )
    parser.add_argument(
        '--input_prefix',
        type=str,
        default='inference/input',
        help='S3 prefix for input data (default: "inference/input")'
    )
    parser.add_argument(
        '--output_prefix',
        type=str,
        default='inference/output',
        help='S3 prefix for output data (default: "inference/output")'
    )

    # Model configurations
    parser.add_argument(
        '--model_paths',
        type=str,
        required=True,
        help='JSON input of model paths'
    )
    parser.add_argument(
        '--framework_version',
        type=str,
        default='2.1',
        help='PyTorch framework version (default: "2.1")'
    )
    parser.add_argument(
        '--py_version',
        type=str,
        default='py310',
        help='Python version for the framework (default: "py310")'
    )

    # Transformer configurations
    parser.add_argument(
        '--instance_count',
        type=int,
        default=1,
        help='Number of instances for the transformer (default: 1)'
    )
    parser.add_argument(
        '--instance_type',
        type=str,
        default='ml.g5.2xlarge',
        help='Instance type for the transformer (default: "ml.g5.2xlarge")'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='MultiRecord',
        help='Transform strategy (default: "MultiRecord")'
    )
    parser.add_argument(
        '--max_concurrent_transforms',
        type=int,
        default=10,
        help='Maximum concurrent transforms (default: 10)'
    )
    parser.add_argument(
        '--max_payload',
        type=int,
        default=10,
        help='Maximum payload size (default: 10)'
    )

    # Data configurations
    parser.add_argument(
        '--content_type',
        type=str,
        default='application/jsonlines',
        help='Content type of the input data (default: "application/jsonlines")'
    )
    parser.add_argument(
        '--split_type',
        type=str,
        default='None',
        help='How to split the input data (default: "None")'
    )

    # Model naming
    parser.add_argument(
        '--model_name_prefix',
        type=str,
        default='tf-binding',
        help='Prefix for the model name (default: "tf-binding")'
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Parse the model artifact locations from input
    args.model_artifact_s3_locations = parse_model_paths(args.model_paths)

    main(args)