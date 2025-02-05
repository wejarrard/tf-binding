import boto3
import os
import argparse
import json
import pandas as pd
from sagemaker import Session
from sagemaker.pytorch import PyTorchModel
import time
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import sys
import logging
from tqdm import tqdm
import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def setup_aws_clients() -> Tuple[Session, boto3.client]:
    """Initialize AWS and SageMaker sessions"""
    logger.info("Initializing AWS and SageMaker sessions")
    session = Session()
    s3_client = boto3.client('s3')
    return session, s3_client

def clean_s3_path(s3_client: Any, bucket: str, prefix: str) -> None:
    """Clean existing files from S3 path"""
    logger.info(f"Cleaning S3 path: s3://{bucket}/{prefix}")
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' in response:
        for item in response['Contents']:
            s3_client.delete_object(Bucket=bucket, Key=item['Key'])
        logger.info(f"Cleaned up {bucket}/{prefix}")

def ensure_clean_directory(directory: str) -> None:
    """Ensure directory exists and is empty"""
    logger.info(f"Ensuring directory is clean: {directory}")
    dir_path = Path(directory)
    if dir_path.exists():
        for file_path in dir_path.glob('*'):
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"Could not delete {file_path}: {e}")
    else:
        dir_path.mkdir(parents=True)
        logger.info(f"Created directory: {directory}")

def create_job_name(cell_line: str, sample: str) -> str:
    """Create unique job name with timestamp"""
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
    cell_line = cell_line.replace('_', '-')
    sample = sample.replace('_', '-')
    job_name = f"{cell_line}-{sample}-{timestamp}"
    logger.info(f"Created job name: {job_name}")
    return job_name

def download_s3_files(
    s3_client: Any,
    bucket: str,
    prefix: str,
    local_dir: str,
    max_retries: int = 3
) -> bool:
    """Download files from S3 with retry logic"""
    logger.info(f"Downloading files from s3://{bucket}/{prefix} to {local_dir}")
    for attempt in range(max_retries):
        try:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            if 'Contents' not in response:
                logger.warning(f"No files found in s3://{bucket}/{prefix}")
                return False

            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('/'):
                    continue
                    
                local_path = Path(local_dir) / Path(key).name
                s3_client.download_file(bucket, key, str(local_path))
                logger.info(f'Downloaded {key} to {local_path}')
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Attempt {attempt + 1} failed. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to download after {max_retries} attempts: {e}")
                return False

def process_jsonl_files(directory: str) -> Optional[pl.LazyFrame]:
    """Combine JSONL files into a LazyFrame using streaming"""
    logger.info(f"Processing JSONL files in {directory}")
    try:
        json_files = list(Path(directory).glob('*.jsonl.gz.out'))
        if not json_files:
            logger.warning(f"No JSONL files found in {directory}")
            return None

        # Read and collect valid LazyFrames
        lazy_frames: List[pl.LazyFrame] = []
        for file in json_files:
            try:
                lf = pl.read_json(file).lazy()
                lazy_frames.append(lf)
                logger.info(f"Successfully processed {file}")
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                continue

        if not lazy_frames:
            logger.warning("No valid LazyFrames created from JSONL files")
            return None

        # Combine all lazy frames
        result_lf = pl.concat(lazy_frames, how="vertical")
        logger.info(f"Created LazyFrame from {len(lazy_frames)} files")
        return result_lf

    except Exception as e:
        logger.error(f"Error processing JSONL files: {e}")
        return None

def create_pytorch_model(
    job_name: str,
    model_path: str,
    args: Any,
    sagemaker_session: Session
) -> PyTorchModel:
    """Create and configure PyTorch model"""
    logger.info(f"Creating PyTorch model with path: {model_path}")
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
            "TS_MAX_RESPONSE_SIZE": "500000000",
            "TS_DEFAULT_STARTUP_TIMEOUT": "1000",
            'TS_DEFAULT_RESPONSE_TIMEOUT': '10000',
            "SAGEMAKER_MODEL_SERVER_WORKERS": "2"
        }
    )

def run_transform_job(
    transformer: Any,
    input_data: str,
    job_name: str,
    args: Any
) -> None:
    """Run transformation job"""
    logger.info(f"Starting transform job: {job_name}")
    transformer.transform(
        data=input_data,
        data_type="S3Prefix",
        content_type=args.content_type,
        split_type=args.split_type,
        job_name=job_name
    )
def save_and_process_results(
    job_name: str,
    args: Any,
    s3_client: Any
) -> Optional[pl.LazyFrame]:
    """Download and process results after job completion"""
    logger.info(f"Processing results for job: {job_name}")
    logger.info(f"All arguments: {args}")

    output_dir = f"{args.project_path}/data/jsonl_output/{args.model}-{args.sample}"
    ensure_clean_directory(output_dir)
    
    s3_prefix = f"inference/output/{job_name}/"
    download_s3_files(s3_client, args.s3_bucket, s3_prefix, output_dir)
    
    lazy_frame = process_jsonl_files(output_dir)
    if lazy_frame is None:
        return None
        
    output_file = f"{args.project_path}/data/processed_results/{args.model}_{args.sample}_processed.parquet"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Materialize and save to parquet
        df = lazy_frame.collect()
        df.write_parquet(
            output_file,
            compression="snappy",
            row_group_size=100_000
        )
        logger.info(f"Results saved to {output_file}")
        return lazy_frame
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return None

def run_inference_pipeline(args: Any) -> Optional[pl.LazyFrame]:
    """Run complete inference pipeline including download and processing"""
    logger.info("Starting inference pipeline")
    try:
        # Initialize AWS clients
        sagemaker_session, s3_client = setup_aws_clients()
        
        # Create job name
        job_name = create_job_name(args.model, args.sample)
        
        # Clean S3 paths
        for prefix in [f"{args.input_prefix}/{job_name}", f"{args.output_prefix}/{job_name}"]:
            clean_s3_path(s3_client, args.s3_bucket, prefix)
        
        # Upload input data
        logger.info("Uploading input data to S3")
        inputs = sagemaker_session.upload_data(
            path=args.local_dir,
            bucket=args.s3_bucket,
            key_prefix=f"{args.input_prefix}/{job_name}"
        )
        
        # Load model path from JSON
        logger.info(f"Loading model paths from {args.model_paths_file}")
        with open(args.model_paths_file) as f:
            model_paths = json.load(f)
            
        if args.model not in model_paths:
            logger.error(f"Model {args.model} not found in paths file")
            raise ValueError(f"Model {args.model} not found in paths file")
            
        model_path = model_paths[args.model]
        
        # Create and configure model
        pytorch_model = create_pytorch_model(job_name, model_path, args, sagemaker_session)
        
        # Configure transformer
        logger.info("Configuring transformer")
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
        logger.info(f"Started job: {job_name}")
        
        # Process and save results
        return save_and_process_results(job_name, args, s3_client)
        
    except Exception as e:
        logger.error(f"Error in inference pipeline: {str(e)}")
        return None

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
        logger.info("Starting main execution")
        df = run_inference_pipeline(args)
        if df is not None:
            logger.info("Pipeline completed successfully")
            sys.exit(0)
        else:
            logger.error("Pipeline failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        sys.exit(1)
    # job_name = create_job_name(args.model, args.sample)
    # sagemaker_session, s3_client = setup_aws_clients()
    # save_and_process_results("NEUROD1-chr3-SRR12455432-2024-12-23-07-28-47", args, s3_client)