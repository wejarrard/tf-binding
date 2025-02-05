#!/usr/bin/env python3
# WAJ 2024-12-03

import sys
import logging
import pickle
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
from captum.attr import DeepLift
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# Add project root to path
root_dir = "/home/ec2-user/projects/tf-binding/src/interpretability"
if root_dir not in sys.path:
    sys.path.append(root_dir)

from scripts import utils as tfb
from scripts.baselines import BaselineGenerator

class AttributionPipeline:
    def __init__(self, model_name: str, output_dir: str, batch_size: int = 1,
                 device: str = None, atac_method: str = 'random', 
                 smooth_sigma: float = 2.0, draw_heatmap:bool=False):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.atac_method = atac_method
        self.smooth_sigma = smooth_sigma
        self.logger = self._setup_logger()
        self.paths = self._setup_paths()
        self.attributions = []
        self.draw_heatmap = draw_heatmap
        
        # Load and setup model
        self.model = tfb.model_fn(str(self.paths['model']), device=self.device, logger=self.logger)
        self.model.eval()
        self.dl = DeepLift(self.model)
    
    def _setup_logger(self, log_level=logging.INFO):
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger
    
    def _setup_paths(self) -> Dict[str, Path]:
        paths = {
            'output': Path(root_dir) / self.output_dir,
            'model': Path(root_dir) / 'models' / self.model_name,
            'data': Path(root_dir) / 'data'
        }
        paths['output'].mkdir(parents=True, exist_ok=True)
        return paths
    
    def process_batch(self, inputs):
        """Process a single batch through DeepLift"""
        inputs = inputs.to(self.device)
        generator = BaselineGenerator(
            seq_length=inputs.shape[1],
            n_channels=inputs.shape[2],
            device=self.device,
        )
        
        baselines = generator.generate_baselines(
            inputs,
            atac_method=self.atac_method,
            smooth_sigma=self.smooth_sigma
        )
        
        with torch.no_grad():
            attributions = self.dl.attribute(inputs, baselines=baselines)
        return attributions.cpu().numpy()
    
    def process_json_file(self, json_path: Path) -> List:
        """Process a single JSON file and compute attributions"""
        with open(json_path, "rb") as f:
            request_body = f.read()
        
        dataset = tfb.input_fn(request_body, "application/jsonlines", logger=self.logger)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True,
        )
        
        file_attributes = []
        for batch in dataloader:
            inputs = batch["input"]
            attributions = self.process_batch(inputs)
            file_attributes.append(attributions)
        
        return file_attributes
    
    def compute_position_features_stats(self) -> pd.DataFrame:
        """Compute statistics across positions and features"""
        # Combine all attributions and reshape
        # attrs_array = np.concatenate(self.attributions, axis=0).squeeze(axis=1)
        attrs_array = np.concatenate(self.attributions, axis=0)
        # Ensure we have shape (samples, seq_len, features)
        if len(attrs_array.shape) == 4: # If shape is (samples, 1, seq_len, features)
            attrs_array = attrs_array.squeeze(axis=1)
        seq_len, n_features = attrs_array.shape[1:]
        
        stats_data = []
        for pos in range(seq_len):
            for feat in range(n_features):
                pos_feat_values = attrs_array[:, pos, feat]
                stats_data.append({
                    'position': pos,
                    'feature': feat,
                    'mean': np.mean(pos_feat_values),
                    'std': np.std(pos_feat_values),
                    'abs_mean': np.mean(np.abs(pos_feat_values)),
                    'max': np.max(pos_feat_values),
                    'min': np.min(pos_feat_values)
                })
        
        return pd.DataFrame(stats_data)
    
    def find_significant_positions(self,
                                 pvalue_threshold: float = 0.05,
                                 effect_size_threshold: float = 0.1) -> pd.DataFrame:
        """Find positions with statistically significant attributions"""
        # attrs_array = np.concatenate(self.attributions, axis=0).squeeze(axis=1)
        attrs_array = np.concatenate(self.attributions, axis=0)
        if len(attrs_array.shape) == 4:
            attrs_array = attrs_array.squeeze(axis=1)
        seq_len, n_features = attrs_array.shape[1:]
        
        sig_data = []
        for pos in range(seq_len):
            for feat in range(n_features):
                values = attrs_array[:, pos, feat]
                t_stat, p_value = stats.ttest_1samp(values, 0)
                effect_size = np.mean(values) / (np.std(values) + 1e-10)
                
                sig_data.append({
                    'position': pos,
                    'feature': feat,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'significant': (p_value < pvalue_threshold) and (abs(effect_size) > effect_size_threshold)
                })
        
        return pd.DataFrame(sig_data)
    
    def plot_attribution_heatmap(self, output_path: str) -> None:
        """Generate heatmap of attribution scores"""
        # attrs_array = np.concatenate(self.attributions, axis=0).squeeze(axis=1)
        attrs_array = np.concatenate(self.attributions, axis=0)
        if len(attrs_array.shape) == 4:
            attrs_array = attrs_array.squeeze(axis=1)
        mean_attrs = np.mean(attrs_array, axis=0)  # Average across samples
        
        plt.figure(figsize=(16, 9))
        sns.heatmap(mean_attrs,
                   cmap='RdBu_r',
                   center=0,
                   cbar_kws={'label': 'Mean Attribution Score'})
        plt.title(f'Attribution Scores for {self.model_name}')
        plt.xlabel('Feature')
        plt.ylabel('Sequence Position')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def run_pipeline(self):
        """Run the complete attribution pipeline"""
        # Process all JSON files
        json_files = sorted(self.paths['data'].glob('jsonl/*.jsonl.gz'))
        self.logger.info(f"Found {len(json_files)} JSON files to process")
        
        for json_file in tqdm(json_files):
            file_attributes = self.process_json_file(json_file)
            self.attributions.extend(file_attributes)
        
        # Generate timestamp for output files
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
        
        # Compute statistics and significant positions
        stats_df = self.compute_position_features_stats()
        significant_df = self.find_significant_positions()
        
        # Save heatmap
        if self.draw_heatmap:
            self.plot_attribution_heatmap(self.paths['output'] / f'{self.model_name}_heatmap_{timestamp}.png')
        
        # Generate summary text
        sig_positions = significant_df[significant_df['significant']].groupby('position').size()
        summary_text = f"""Model: {self.model_name}
Data shape: {np.concatenate(self.attributions, axis=0).shape}

Significant Positions Summary:
Total positions with significant features: {len(sig_positions)}

Top 10 positions by number of significant features:
{sig_positions.nlargest(10).to_string()}

Top 10 position-feature pairs by absolute mean attribution:
{stats_df.nlargest(10, 'abs_mean')[['position', 'feature', 'abs_mean', 'mean']].to_string()}
"""
        
        # Save all outputs
        outputs = {
            'attributions': self.attributions,
            'stats': stats_df,
            'significant_positions': significant_df,
            'summary': summary_text
        }
        
        with open(self.paths['output'] / f'{self.model_name}_results_{timestamp}.pkl', 'wb') as f:
            pickle.dump(outputs, f)
        
        self.logger.info(f"Pipeline completed. Results saved to {self.paths['output']}")

def main():
    parser = argparse.ArgumentParser(description='DeepLift attribution analysis pipeline')
    parser.add_argument('--model_name', type=str, default='AR-Full',
                       help='Name of the model to analyze (default: AR-Full).')
    parser.add_argument('--output_dir', type=str, default='DLS_output',
                       help='Directory for output files (default: DLS_output).')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing (default: 1).')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                       help='Device to use (default: auto-detect).')
    parser.add_argument('--atac_method', default='random',
                       help="Method for creating ATAC channel's DeepLift Baseline (default: random).")
    parser.add_argument('--smooth_sigma', type=float, default=2.0,
                        help="Standard deviation for Gaussian smoothing of shuffled ATAC signal (default: 2.0). Higher values yield broader smoothing. Typical range 1-5.")
    args = parser.parse_args()
    
    pipeline = AttributionPipeline(
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        atac_method=args.atac_method,
        smooth_sigma=args.smooth_sigma,
        draw_heatmap=False,
    )
    
    pipeline.run_pipeline()

if __name__ == '__main__':
    main()