import boto3
import os
import argparse
import json
from sagemaker import Session
from sagemaker.pytorch import PyTorchModel
import time

# Initialize a SageMaker session
sagemaker_session = Session()

role = "arn:aws:iam::016114370410:role/tf-binding-sites"
local_dir = "/Users/wejarrard/projects/tf-binding/data/jsonl"
s3 = boto3.client('s3')

# Function to delete all objects in a specified S3 bucket/prefix
def delete_s3_objects(bucket_name, prefix=""):
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if 'Contents' in response:
        for item in response['Contents']:
            s3.delete_object(Bucket=bucket_name, Key=item['Key'])
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

if __name__ == "__main__":
    # Set up argument parser to accept model paths as a JSON input string
    parser = argparse.ArgumentParser(description='Run SageMaker transform jobs with specified model paths.')
    parser.add_argument('--model_paths', type=str, required=True, help='JSON input of model paths')

    # Parse the command line arguments
    args = parser.parse_args()

    # Parse the model artifact locations from input
    model_artifact_s3_locations = parse_model_paths(args.model_paths)

    # Delete existing files from the specified S3 locations
    delete_s3_objects(bucket_name="tf-binding-sites", prefix="inference/input")
    for key in model_artifact_s3_locations:
        delete_s3_objects(bucket_name="tf-binding-sites", prefix=f"inference/output/{key}")

    # Upload new files to the specified S3 location
    inputs = sagemaker_session.upload_data(path=local_dir, bucket="tf-binding-sites", key_prefix="inference/input")
    print(f"Input spec: {inputs}")

    # Create PyTorchModel and run transform jobs
    for cell_line_name, model_artifact_s3_location in model_artifact_s3_locations.items():
        cell_line_name = f"{cell_line_name}-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        pytorch_model = PyTorchModel(
            model_data=model_artifact_s3_location,
            role=role,
            framework_version="2.1",
            py_version="py310",
            source_dir="/Users/wejarrard/projects/tf-binding/src/inference/scripts",
            entry_point="inference.py",
            sagemaker_session=sagemaker_session,
            name = f"tf-binding-{cell_line_name}"
        )

        # Create transformer from PyTorchModel object
        output_path = f"s3://tf-binding-sites/inference/output/{cell_line_name}"
        transformer = pytorch_model.transformer(
            instance_count=1, 
            instance_type="ml.g5.2xlarge", 
            output_path=output_path,
            strategy="MultiRecord",
            max_concurrent_transforms=10,
            max_payload=10,
        )

        # Start the transform job
        transformer.transform(
            data=inputs,
            data_type="S3Prefix",
            content_type="application/jsonlines",
            split_type="None",
            wait=False,
            job_name=f"{cell_line_name}"
        )

        print(f"Transformation output saved to: {output_path}")