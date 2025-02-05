import boto3
import os
import pandas as pd
from pathlib import Path
from sagemaker import Session
import json
from typing import Optional, Tuple
import time
from typing import Any

def setup_aws_clients() -> Tuple[Session, boto3.client]:
    """Initialize AWS and SageMaker sessions"""
    return Session(), boto3.client('s3')

def ensure_clean_directory(directory: str) -> None:
    """Ensure directory exists and is empty"""
    dir_path = Path(directory)
    if dir_path.exists():
        for file_path in dir_path.glob('*'):
            try:
                file_path.unlink()
            except Exception as e:
                print(f"Warning: Could not delete {file_path}: {e}")
    else:
        dir_path.mkdir(parents=True)

def download_s3_files(
    s3_client: Any,
    bucket: str,
    prefix: str,
    local_dir: str,
    max_retries: int = 3
) -> bool:
    """Download all files from S3 prefix with retry logic"""
    for attempt in range(max_retries):
        try:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            if 'Contents' not in response:
                print(f"No files found in s3://{bucket}/{prefix}")
                return False

            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('/'):
                    continue
                    
                local_path = Path(local_dir) / Path(key).name
                s3_client.download_file(bucket, key, str(local_path))
                print(f'Downloaded {key} to {local_path}')
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Attempt {attempt + 1} failed. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download after {max_retries} attempts: {e}")
                return False

def process_jsonl_files(directory: str) -> Optional[pd.DataFrame]:
    """Combine all JSONL files into a single DataFrame"""
    try:
        json_files = list(Path(directory).glob('*.jsonl.gz.out'))
        if not json_files:
            print(f"No JSONL files found in {directory}")
            return None

        dfs = []
        for file in json_files:
            try:
                df = pd.read_json(file)
                dfs.append(df)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

        if not dfs:
            return None

        return pd.concat(dfs, ignore_index=True)

    except Exception as e:
        print(f"Error processing JSONL files: {e}")
        return None

def get_job_name(base_dir: str, model_name: str, cell_line: str) -> Optional[str]:
    """Get job name from saved JSON file"""
    try:
        job_names_file = os.path.join(base_dir, "data", "job_names.json")
        if not os.path.exists(job_names_file):
            print(f"Job names file not found: {job_names_file}")
            return None
            
        with open(job_names_file) as f:
            job_names = json.load(f)
            
        key = f"{model_name}-{cell_line}"
        if key not in job_names:
            print(f"No job name found for {key}")
            return None
            
        return job_names[key]
        
    except Exception as e:
        print(f"Error reading job names: {e}")
        return None

def process_s3_data(model_name: str, cell_line: str) -> Optional[pd.DataFrame]:
    """Main function to process S3 data"""
    base_dir = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"
    local_dir = f"{base_dir}/data/jsonl_output/{model_name}-{cell_line}"
    
    # Get job name
    job_name = get_job_name(base_dir, model_name, cell_line)
    if not job_name:
        return None
        
    s3_prefix = f"inference/output/{job_name}/"
    bucket_name = "tf-binding-sites"
    
    # Initialize AWS clients
    _, s3_client = setup_aws_clients()
    
    # Prepare local directory
    ensure_clean_directory(local_dir)
    
    # Download and process files
    if download_s3_files(s3_client, bucket_name, s3_prefix, local_dir):
        return process_jsonl_files(local_dir)
    return None

def save_results(df: pd.DataFrame, model_name: str, cell_line: str) -> None:
    """Save processed results to CSV"""
    output_file = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/processed_results/{model_name}_{cell_line}_processed.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False, header=True, sep='\t')
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python download_results.py MODEL_NAME CELL_LINE")
        sys.exit(1)
        
    MODEL_NAME = sys.argv[1]
    CELL_LINE = sys.argv[2]
    
    df = process_s3_data(MODEL_NAME, CELL_LINE)
    if df is not None:
        save_results(df, MODEL_NAME, CELL_LINE)