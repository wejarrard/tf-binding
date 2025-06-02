import os
from typing import Any, Optional, List
import polars as pl
from pathlib import Path



def process_jsonl_files(directory: str) -> Optional[pl.LazyFrame]:
    """Combine JSONL files into a LazyFrame using streaming"""
    print(f"Processing JSONL files in {directory}")
    try:
        json_files = list(Path(directory).glob('*.jsonl.gz.out'))
        if not json_files:
            print(f"No JSONL files found in {directory}")
            return None

        # Read and collect valid LazyFrames
        lazy_frames: List[pl.LazyFrame] = []
        for file in json_files:
            try:
                lf = pl.read_json(file).lazy()
                lazy_frames.append(lf)
                print(f"Successfully processed {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

        if not lazy_frames:
            print("No valid LazyFrames created from JSONL files")
            return None

        # Combine all lazy frames
        result_lf = pl.concat(lazy_frames, how="vertical")
        print(f"Created LazyFrame from {len(lazy_frames)} files")
        return result_lf

    except Exception as e:
        print(f"Error processing JSONL files: {e}")
        return None

def save_and_process_results(
    project_path: str,
    model: str,
    sample: str
) -> Optional[pl.LazyFrame]:
    """Download and process results after job completion"""

    output_dir = f"{project_path}/data/jsonl_output/{model}-{sample}"
    
    
    lazy_frame = process_jsonl_files(output_dir)
    if lazy_frame is None:
        return None
        
    output_file = f"{project_path}/data/processed_results/{model}_{sample}_processed.parquet"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Materialize and save to parquet
        df = lazy_frame.collect()
        df.write_parquet(
            output_file,
            compression="snappy",
            row_group_size=100_000
        )
        print(f"Results saved to {output_file}")
        return lazy_frame
    except Exception as e:
        print(f"Error saving results: {e}")
        return None
    
def main():
    project_path = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"
    model = "ERG-CellLine"
    sample = "VCAP"
    save_and_process_results(project_path, model, sample)

if __name__ == "__main__":
    main()