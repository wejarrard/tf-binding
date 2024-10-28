#!/usr/bin/env python3

import argparse
import subprocess
import os
import boto3
from sagemaker import Session
from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig

def main():
    # Parse command-line arguments with default values
    parser = argparse.ArgumentParser(description='Run TF Binding Site Training Script')
    parser.add_argument('--tf_name', required=True, help='Name of the transcription factor (default: "AR")')
    parser.add_argument('--cell_line', required=True, help='Name of the cell line (default: "22Rv1")')
    parser.add_argument('--path_to_project', default='/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding', help='Path to the project directory')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for the training job (default: 1e-5)')

    # parser.add_argument('--local_dir', default='/Users/wejarrard/projects/tf-binding/data/data_splits', help='Local directory for data splits')
    parser.add_argument('--role', default='arn:aws:iam::016114370410:role/tf-binding-sites', help='AWS IAM role ARN')
    parser.add_argument('--s3_bucket', default='tf-binding-sites', help='S3 bucket name (default: "tf-binding-sites")')
    parser.add_argument('--s3_prefix', default='pretraining/data/', help='S3 prefix for data (default: "pretraining/data/")')
    parser.add_argument('--entry_point', default='multi_tf_prediction.py', help='Path to the entry point script for training')
    parser.add_argument('--source_dir', default='../training', help='Path to the source directory for training scripts')
    parser.add_argument('--instance_type', default='ml.g5.8xlarge', help='SageMaker instance type (default: "ml.g5.8xlarge")')
    args = parser.parse_args()
    
    # Assign arguments to variables
    TF_NAME = args.tf_name
    CELL_LINE = args.cell_line
    path_to_project = args.path_to_project
    local_dir = os.path.join(path_to_project, 'data', 'data_splits')
    role = args.role
    s3_bucket = args.s3_bucket
    s3_prefix = args.s3_prefix
    entry_point = args.entry_point
    source_dir = args.source_dir
    instance_type = args.instance_type
    learning_rate = args.learning_rate
    
    # Step 1: Generate training peaks
    print("Generating training peaks...")
    scripts_dir = os.path.join(path_to_project, 'scripts')
    os.chdir(scripts_dir)
    generate_peaks_script = os.path.join(path_to_project, 'src', 'processing', 'generate_training_peaks.py')
    subprocess.run([
        'python',
        generate_peaks_script,
        TF_NAME,
        '--balance',
        '--validation_cell_lines',
        CELL_LINE
    ])
    
    # Step 2: Upload data to S3
    print("Uploading data to S3...")
    sagemaker_session = Session(default_bucket=s3_bucket)
    inputs = sagemaker_session.upload_data(path=local_dir, key_prefix=s3_prefix)
    print(f"Data uploaded to S3 at: {inputs}")
    
    # Step 3: Set up SageMaker training job
    print("Setting up SageMaker training job...")
    output_s3_path = f"s3://{s3_bucket}/finetuning/results/output"
    tensorboard_output_config = TensorBoardOutputConfig(
        s3_output_path=f"s3://{s3_bucket}/finetuning/results/tensorboard",
        container_local_output_path="/opt/ml/output/tensorboard"
    )
    
    estimator = PyTorch(
        base_job_name=f"{TF_NAME}-Full-Data-Model",
        entry_point=entry_point,
        source_dir=source_dir,
        output_path=output_s3_path,
        code_location=f"s3://{s3_bucket}/finetuning/results/code",
        role=role,
        py_version="py310",
        framework_version='2.0.0',
        volume_size=600,
        instance_count=1,
        max_run=1209600,
        instance_type=instance_type,
        hyperparameters={
            'learning-rate': learning_rate
        },
        tensorboard_output_config=tensorboard_output_config,
    )
    
    training_data_s3_path = f"s3://{s3_bucket}/{s3_prefix}"
    estimator.fit({'training': training_data_s3_path}, wait=False)
    print("SageMaker training job initiated.")
    
if __name__ == '__main__':
    main()