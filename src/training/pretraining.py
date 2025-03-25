#!/usr/bin/env python3

import argparse
import subprocess
import os
import boto3
from sagemaker import Session
from sagemaker.pytorch import PyTorch
from tf_finetuning.tf_dataloader import TransformType, FilterType

def main():
    parser = argparse.ArgumentParser(description='Run TF Binding Site Training Script')
    parser.add_argument('--tf_name', required=True, help='Name of the transcription factor')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--cell_line', type=str, help='Name of the cell line for validation set')
    group.add_argument('--chr', type=str, help='Chromosome(s) for validation set (space-separated)')
    group.add_argument('--random', action='store_true', help='randomly select 15% of the data for validation set')
    
    parser.add_argument('--negative_regions_bed', type=str, nargs='+', help='Paths to BED files containing negative regions')
    parser.add_argument('--path_to_project', default='/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--role', default='arn:aws:iam::016114370410:role/tf-binding-sites')
    parser.add_argument('--s3_bucket', default='tf-binding-sites')
    parser.add_argument('--s3_prefix', default='pretraining/data')
    parser.add_argument('--entry_point', default='tf_prediction.py')
    parser.add_argument('--instance_type', default='ml.g5.8xlarge')
    parser.add_argument('--transform_type', default='none')
    parser.add_argument('--filter_type', default='none')
    args = parser.parse_args()
    
    # Generate peaks command
    command = [
        'python',
        os.path.join(args.path_to_project, 'src', 'utils', 'generate_training_peaks.py'),
        '--tf', args.tf_name,
        '--balance'
    ]

    if args.cell_line:
        command.extend(['--validation_cell_lines', args.cell_line])
    elif args.chr:
        command.extend(['--validation_chromosomes'] + args.chr.split())
    elif args.random:
        pass
        
    if args.negative_regions_bed:
        command.extend(['--negative_regions_bed'] + args.negative_regions_bed)

    # Execute peak generation
    print("Generating training peaks...")
    os.chdir(args.path_to_project)
    subprocess.run(command, check=True)

    # Rename files
    local_dir = os.path.join(args.path_to_project, 'data', 'data_splits')
    validation_identifier = args.tf_name + '-' + (args.cell_line if args.cell_line else args.chr if args.chr else 'random')
    for file_type in ['training', 'validation']:
        old_name = os.path.join(local_dir, f'{file_type}_combined.csv')
        new_name = os.path.join(local_dir, f'{file_type}_combined_{validation_identifier}.csv')
        
        if not os.path.exists(old_name):
            continue
            
        # Check if new file exists and compare contents
        if os.path.exists(new_name):
            with open(old_name, 'rb') as f1, open(new_name, 'rb') as f2:
                if f1.read() == f2.read():
                    os.remove(old_name)  # Remove the old file since it's identical
                    continue
        
        # Either new file doesn't exist or contents are different
        if os.path.exists(new_name):
            os.remove(new_name)  # Remove existing file before renaming
        os.rename(old_name, new_name)

    # Upload to S3
    print("Uploading data to S3...")
    sagemaker_session = Session()
    inputs = sagemaker_session.upload_data(path=local_dir, bucket=args.s3_bucket, key_prefix=args.s3_prefix)

    # Configure SageMaker training
    print("Setting up SageMaker training job...")
    estimator = PyTorch(
        base_job_name="pretraining-unfrozen-transformer",
        entry_point="pretrain.py",
        model_dir='/opt/ml/model',
        source_dir="./pretraining",
        output_path="s3://tf-binding-sites/pretraining/output",
        py_version="py310",
        framework_version='2.0.0',
        volume_size=800,
        max_run=1209600,
        instance_count=3,
        instance_type='ml.g5.12xlarge',
        hyperparameters={
            'learning-rate': 1e-4
        },
        distribution=distribution,
        sagemaker_session=sagemaker_session
    )
    
    estimator.fit({'training': inputs}, wait=False)
    print("Training job initiated")

if __name__ == '__main__':
    main()