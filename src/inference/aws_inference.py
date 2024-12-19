import boto3
import os
import argparse
import json
from sagemaker import Session
from sagemaker.pytorch import PyTorchModel
import time
from typing import Dict, Any
from pathlib import Path
import sys

def setup_s3_client():
    """Initialize S3 client"""
    return boto3.client('s3')

def clean_s3_path(s3_client: Any, bucket: str, prefix: str) -> None:
    """Clean existing files from S3 path"""
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' in response:
        for item in response['Contents']:
            s3_client.delete_object(Bucket=bucket, Key=item['Key'])
        print(f"Cleaned up {bucket}/{prefix}")

def create_job_name(cell_line: str, sample: str) -> str:
    """Create unique job name with timestamp"""
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
    return f"{cell_line}-{sample}-{timestamp}"

def create_pytorch_model(
    job_name: str,
    model_path: str,
    args: Any,
    sagemaker_session: Session
) -> PyTorchModel:
    """Create and configure PyTorch model"""
    return PyTorchModel(
        model_data=model_path,
        role=args.iam_role,
        framework_version=args.framework_version,
        py_version=args.py_version,
        source_dir=os.path.join(args.project_path, args.source_dir),
        entry_point=args.entry_point,
        sagemaker_session=sagemaker_session,
        name=f"{args.model_name_prefix}-{job_name}",
        env={
            "TS_MAX_RESPONSE_SIZE": "100000000",
            "TS_DEFAULT_STARTUP_TIMEOUT": "600",
            'TS_DEFAULT_RESPONSE_TIMEOUT': '1000',
            "SAGEMAKER_MODEL_SERVER_WORKERS": "4"
        }
    )

def run_transform_job(
    transformer: Any,
    input_data: str,
    job_name: str,
    args: Any
) -> None:
    """Run transformation job"""
    transformer.transform(
        data=input_data,
        data_type="S3Prefix",
        content_type=args.content_type,
        split_type=args.split_type,
        job_name=job_name
    )

def save_job_names(job_names: Dict[str, str], output_path: str) -> None:
    """Save job names to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(job_names, f, indent=2)
    print(f"Job names saved to: {output_path}")


def load_model_paths(json_path: str, model: str) -> Dict[str, str]:
    """
    Load and validate model path from JSON file for specified model
    
    Args:
        json_path: Path to JSON file containing model paths
        model: Name of model to load path for
        
    Returns:
        Dict with single model name and path
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON content is invalid or model not found
    """
    if not Path(json_path).exists():
        raise FileNotFoundError(f"Model paths JSON file not found: {json_path}")
        
    try:
        with open(json_path) as f:
            model_paths = json.load(f)
            
        if not isinstance(model_paths, dict):
            raise ValueError("Model paths must be a JSON object")
            
        if model not in model_paths:
            raise ValueError(f"Model {model} not found in paths file")
            
        path = model_paths[model]
        # Remove path existence check since S3 paths can't be validated locally
        return {model: path}
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {json_path}: {str(e)}")
    

def run_aws_inference_jobs(args: Any) -> Dict[str, str]:
    """Main function to run AWS inference jobs with improved error handling"""
    try:
        # Load and validate model paths from file
        model_paths = load_model_paths(args.model_paths_file, args.model)
            
        sagemaker_session = Session()
        s3 = setup_s3_client()
        job_names = {}
        
        for cell_line_name, model_path in model_paths.items():
            # Create unique job name
            job_name = create_job_name(cell_line_name, args.sample)
            job_names[cell_line_name] = job_name
            
            try:
                # Clean S3 paths
                for prefix in [f"{args.input_prefix}/{job_name}", f"{args.output_prefix}/{job_name}"]:
                    clean_s3_path(s3, args.s3_bucket, prefix)
                
                # Upload input data
                inputs = sagemaker_session.upload_data(
                    path=args.local_dir,
                    bucket=args.s3_bucket,
                    key_prefix=f"{args.input_prefix}/{job_name}"
                )
                
                # Create and configure model
                pytorch_model = create_pytorch_model(job_name, model_path, args, sagemaker_session)
                
                # Configure transformer
                transformer = pytorch_model.transformer(
                    instance_count=args.instance_count,
                    instance_type=args.instance_type,
                    output_path=f"s3://{args.s3_bucket}/{args.output_prefix}/{job_name}",
                    strategy=args.strategy,
                    max_concurrent_transforms=args.max_concurrent_transforms,
                    max_payload=args.max_payload
                )
                
                # Run transform job
                run_transform_job(transformer, inputs, job_name, args)
                print(f"Started job: {job_name}")
                
            except Exception as e:
                print(f"Error processing {cell_line_name}: {str(e)}")
                continue
                
        return job_names
        
    except Exception as e:
        raise RuntimeError(f"Failed to run inference jobs: {str(e)}")

def get_parser() -> argparse.ArgumentParser:
    """Set up and return argument parser with updated model paths argument"""
    parser = argparse.ArgumentParser(description='Run SageMaker transform jobs with specified model paths.')

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Name of the model to use for inference'
    )

    parser.add_argument(
        '--sample',
        type=str,
        required=True,
        help='Name of the sample to use for inference'
    )

    # Update model paths argument to take file path
    parser.add_argument(
        '--model_paths_file',
        type=str,
        required=True,
        help='Path to JSON file containing model paths'
    )
    
    # Rest of the arguments remain the same
    parser.add_argument(
        '--project_path',
        type=str,
        default="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding",
        help='Root path of the project'
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
        default=5,
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

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    try:
        # Run jobs and save job names
        job_names = run_aws_inference_jobs(args)
        
        # Save job names for downstream processing
        output_file = os.path.join(args.project_path, "data", "job_names.json")
        save_job_names(job_names, output_file)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)