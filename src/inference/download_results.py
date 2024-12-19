import boto3
import os
import pandas as pd
import glob
from pathlib import Path
from sagemaker import Session
import json
def setup_aws_client():
    """Initialize AWS and SageMaker sessions"""
    sagemaker_session = Session()
    s3 = boto3.client('s3')
    return sagemaker_session, s3

def clean_directory(directory):
    """Remove all files in specified directory"""
    for file_path in Path(directory).glob('*'):
        try:
            file_path.unlink()
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def download_from_s3(s3_client, bucket, prefix, local_dir):
    """Download all files from S3 prefix to local directory"""
    # Create directory if it doesn't exist
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # List objects within the specified prefix
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        if 'Contents' not in response:
            print(f"No files found in s3://{bucket}/{prefix}")
            return False
            
        # Download each file
        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('/'):  # Skip directories
                continue
                
            local_path = Path(local_dir) / Path(key).name
            s3_client.download_file(bucket, key, str(local_path))
            print(f'Downloaded {key} to {local_path}')
            
        return True
        
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        return False

def combine_jsonl_files(directory):
    """Combine all JSONL files in directory into single DataFrame"""
    try:
        # Find all JSONL files
        json_files = list(Path(directory).glob('*.jsonl.gz.out'))
        
        if not json_files:
            print(f"No JSONL files found in {directory}")
            return None
            
        # Read and combine all files
        dataframes = [pd.read_json(file) for file in json_files]
        return pd.concat(dataframes, ignore_index=True)
        
    except Exception as e:
        print(f"Error combining JSONL files: {e}")
        return None

def process_s3_data(model_name, cell_line):
    """Main function to process S3 data"""
    # Setup paths and configurations
    base_dir = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"
    local_dir = f"{base_dir}/data/jsonl_output/{model_name}-{cell_line}"
    
    # Read job name from JSON file
    with open(os.path.join(base_dir, "data", "job_names.json")) as f:
        job_names = json.load(f)
    
    job_name = job_names[f"{model_name}-{cell_line}"]
    s3_prefix = f"inference/output/{job_name}/"
    bucket_name = "tf-binding-sites"
    
    # Initialize AWS clients
    sagemaker_session, s3_client = setup_aws_client()
    
    # Clean local directory
    print(f"Cleaning directory: {local_dir}")
    clean_directory(local_dir)
    
    # Download files
    print(f"Downloading files from S3...")
    success = download_from_s3(s3_client, bucket_name, s3_prefix, local_dir)
    if not success:
        return None
        
    # Process files
    print("Combining JSONL files...")
    df = combine_jsonl_files(local_dir)
    
    if df is not None:
        print(f"Successfully processed {len(df)} rows of data")
    
    return df

if __name__ == "__main__":
    import sys
    MODEL_NAME = sys.argv[1]
    CELL_LINE = sys.argv[2]
    
    df = process_s3_data(MODEL_NAME, CELL_LINE)
    
    if df is not None:
        # Save to CSV
        output_file = f"/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/processed_results/{MODEL_NAME}_{CELL_LINE}_processed.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False, header=True, sep='\t')
        print(f"Data saved to {output_file}")