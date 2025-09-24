#!/usr/bin/env python3


import os
import sys
import yaml
import logging
import argparse
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from copy import deepcopy
from datetime import datetime

# Add project root to path (go up two levels from tools/th_k_sweep/)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
# Change working directory to project root for proper imports and file access
os.chdir(project_root)

from utils.loader import dataLoader_test as dataLoader
from utils.loader import get_module
from export import export_detector_homoAdapt_gpu


class ThresholdKSweepExperiment:
    """Systematic experiment for threshold and K parameter sweeping"""
    
    def __init__(self, config_path, base_output_dir=None):
        self.config_path = config_path
        
        # Load base configuration
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
            
        # Extract dataset name from config
        self.dataset_name = self._extract_dataset_name()
        
        # Set default output directory if not provided
        if base_output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_output_dir = f"logs/export/{self.dataset_name}/threshold_k_sweep_{timestamp}"
            
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
            
        # Experiment parameters - extended ranges
        # Threshold range: 0.000075 - 0.025 (logarithmic distribution for good coverage)
        self.threshold_values = [
            0.000075, 0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.0007, 0.001,
            0.0015, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.025
        ]
        
        # K range: 2 - 100 (strategic selection for homography adaptation)
        # Note: K=1 causes issues with homography adaptation, so starting from K=2
        self.k_values = [
            2, 4, 6, 8, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100
        ] 
        
        # Results storage
        self.results_list = []
        self.detailed_results = {}
        
        # Load existing results if they exist
        self._load_existing_results()
        
        # Device setup
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        logging.info(f"Dataset: {self.dataset_name}")
        logging.info(f"Output directory: {self.base_output_dir}")
        logging.info(f"Experiment initialized with {len(self.threshold_values)} thresholds × {len(self.k_values)} K values = {len(self.threshold_values) * len(self.k_values)} total experiments")
        
    def _extract_dataset_name(self):
        """Extract dataset name from the images_path in config"""
        images_path = self.base_config.get('data', {}).get('images_path', '')
        
        # Extract the last part of the path (e.g., 'toy_33' from './datasets/endomapper/toy_33/')
        if images_path:
            # Remove trailing slash and get last component
            dataset_name = Path(images_path.rstrip('/')).name
            if dataset_name:
                return dataset_name
        
    def _load_existing_results(self):
        """Load existing results from JSON file if it exists"""
        json_results_file = self.base_output_dir / "threshold_k_sweep_results.json"
        
        if json_results_file.exists():
            try:
                with open(json_results_file, 'r') as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, list):
                        self.results_list = loaded_data
                        logging.info(f"Loaded {len(self.results_list)} existing results from {json_results_file}")
                    elif isinstance(loaded_data, dict) and 'results' in loaded_data:
                        self.results_list = loaded_data['results']
                        logging.info(f"Loaded {len(self.results_list)} existing results from {json_results_file}")
                    else:
                        logging.warning(f"Unexpected JSON format in {json_results_file}")
            except Exception as e:
                logging.error(f"Failed to load existing results: {e}")
    
        
        # Fallback to dataset field if images_path parsing fails
        dataset_field = self.base_config.get('data', {}).get('dataset', 'unknown_dataset')
        return f"{dataset_field.lower()}_dataset"
        
    def run_experiments(self):
        """Run the full sweep of threshold × K experiments"""
        
        experiment_count = 0
        total_experiments = len(self.threshold_values) * len(self.k_values)
        
        results_list = []
        
        for threshold in self.threshold_values:
            for k_value in self.k_values:
                experiment_count += 1
                
                logging.info(f"\n=== Experiment {experiment_count}/{total_experiments}: threshold={threshold:.4f}, K={k_value} ===")
                
                # Create experiment-specific config
                exp_config = self._create_experiment_config(threshold, k_value)
                
                # Create output directory
                # Use microseconds (×1000000) for consistent naming across full range
                exp_name = f"th{int(threshold*1000000):06d}_K{k_value:03d}"
                exp_output_dir = self.base_output_dir / exp_name
                exp_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Run export (keypoint detection)
                logging.info(f"Running export for {exp_name}...")
                try:
                    self._run_single_export(exp_config, exp_output_dir, exp_name)
                    
                    # Evaluate results  
                    logging.info(f"Evaluating results for {exp_name}...")
                    metrics = self._evaluate_experiment(exp_output_dir, threshold, k_value)
                    
                    # Store results
                    results_list.append(metrics)
                    
                    logging.info(f"✅ Completed {exp_name}: {metrics['num_keypoints']:.1f} keypoints")
                    
                except Exception as e:
                    logging.error(f"❌ Failed {exp_name}: {str(e)}")
                    # Add failed experiment with NaN values
                    failed_metrics = {
                        'threshold': threshold,
                        'k_value': k_value,
                        'experiment_name': exp_name,
                        'num_keypoints': np.nan,
                        'keypoint_std': np.nan,
                        'status': 'failed'
                    }
                    results_list.append(failed_metrics)
        
        # Convert to list and save
        self.results_list = results_list
        self._save_results()
        self._generate_analysis_plots()
        
        logging.info(f"\n🎉 All experiments completed! Results saved to {self.base_output_dir}")
        
    def _create_experiment_config(self, threshold, k_value):
        """Create configuration for specific threshold and K value"""
        config = deepcopy(self.base_config)
        
        # Only update the parameters that vary in the sweep
        config['model']['detection_threshold'] = threshold
        config['data']['homography_adaptation']['num'] = k_value
        config['data']['homography_adaptation']['enable'] = True
        
        return config
        
    def _run_single_export(self, config, output_dir, exp_name):
        """Run keypoint export for a single experiment"""
        
        # Save experiment config
        config_path = output_dir / "config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        # Create args object (mimicking command line interface)
        class Args:
            def __init__(self):
                self.command = "export_detector_homoAdapt"
                self.exper_name = exp_name
                self.debug = False
                self.outputImg = True  # Enable image output for visualization
                self.eval = True       # Enable evaluation
                
        args = Args()
        
        # Run the export function
        export_detector_homoAdapt_gpu(config, str(output_dir), args)
        
    def _evaluate_experiment(self, output_dir, threshold, k_value):
        """Evaluate a single experiment and return metrics"""
        
        # Find exported .npz files
        predictions_dir = output_dir / "predictions" / "train"
        if not predictions_dir.exists():
            raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
            
        npz_files = list(predictions_dir.glob("*.npz"))
        if len(npz_files) == 0:
            raise FileNotFoundError(f"No .npz files found in {predictions_dir}")
            
        logging.info(f"Found {len(npz_files)} prediction files")
        
        # Metrics storage
        keypoint_counts = []
        
        # Process each prediction file
        for npz_file in npz_files[:10]:  # Limit to first 10 files for speed
            try:
                data = np.load(npz_file)
                
                # Basic keypoint count
                if 'pts' in data:
                    pts = data['pts']
                    keypoint_counts.append(len(pts))
                        
            except Exception as e:
                logging.warning(f"Failed to process {npz_file}: {e}")
                
        # Aggregate metrics
        metrics = {
            'threshold': threshold,
            'k_value': k_value, 
            'experiment_name': f"th{int(threshold*1000000):06d}_K{k_value:03d}",
            'num_keypoints': np.mean(keypoint_counts) if keypoint_counts else np.nan,
            'keypoint_std': np.std(keypoint_counts) if keypoint_counts else np.nan,
            'num_processed_files': len(npz_files),
            'status': 'success'
        }
        
        return metrics
        
    def _save_results(self):
        """Save experiment results to files"""
        
        # Save detailed results as JSON
        results_path = self.base_output_dir / "threshold_k_sweep_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results_list, f, indent=2)
        
        # Save as CSV-like format  
        csv_path = self.base_output_dir / "threshold_k_sweep_results.csv"
        with open(csv_path, 'w') as f:
            if self.results_list:
                # Write header
                headers = list(self.results_list[0].keys())
                f.write(','.join(headers) + '\n')
                
                # Write data
                for result in self.results_list:
                    values = [str(result.get(h, '')) for h in headers]
                    f.write(','.join(values) + '\n')
        
        # Save summary statistics
        summary_path = self.base_output_dir / "experiment_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Threshold × K Sweep Experiment Summary\n")
            f.write("=" * 50 + "\n\n")
            
            successful_results = [r for r in self.results_list if r.get('status') == 'success']
            failed_results = [r for r in self.results_list if r.get('status') == 'failed']
            
            f.write(f"Total experiments: {len(self.results_list)}\n")
            f.write(f"Successful experiments: {len(successful_results)}\n")
            f.write(f"Failed experiments: {len(failed_results)}\n\n")
            
            # Best configurations
            if successful_results:
                f.write("Best Configurations:\n")
                f.write("-" * 20 + "\n")
                
                # Highest keypoint count
                keypoint_counts = [r['num_keypoints'] for r in successful_results if not np.isnan(r['num_keypoints'])]
                if keypoint_counts:
                    max_kp_val = max(keypoint_counts)
                    best_kp = next(r for r in successful_results if r['num_keypoints'] == max_kp_val)
                    f.write(f"Most keypoints: {best_kp['experiment_name']} ({best_kp['num_keypoints']:.1f} keypoints)\n")
        
        logging.info(f"Results saved to {results_path}")
        logging.info(f"Summary saved to {summary_path}")
        
    def _generate_analysis_plots(self):
        """Generate analysis plots and heatmaps"""
        
        # Filter successful results
        successful_results = [r for r in self.results_list if r.get('status') == 'success']
        
        if not successful_results:
            logging.warning("No successful results to plot - skipping visualization")
            return
            
        # Set up plotting style
        plt.style.use('default')
        
        # Create plots directory
        plots_dir = self.base_output_dir / "analysis_plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Create only keypoints heatmap
        self._create_keypoints_heatmap(successful_results, plots_dir)
        
        logging.info(f"Analysis plots saved to {plots_dir}")
        
    def _create_keypoints_heatmap(self, successful_results, plots_dir):
        """Create a heatmap showing threshold vs K combinations with keypoint counts"""
        
        if not successful_results:
            return
            
        # Extract unique thresholds and K values
        thresholds = sorted(set([r['threshold'] for r in successful_results]))
        k_values = sorted(set([r['k_value'] for r in successful_results]))
        
        # Create heatmap matrix
        heatmap_data = np.full((len(thresholds), len(k_values)), np.nan)
        
        for result in successful_results:
            if not np.isnan(result['num_keypoints']):
                th_idx = thresholds.index(result['threshold'])
                k_idx = k_values.index(result['k_value'])
                heatmap_data[th_idx, k_idx] = result['num_keypoints']
        
        # Create the heatmap plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Create heatmap
        im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto', origin='lower')
        
        # Set ticks and labels
        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_yticks(range(len(thresholds)))
        ax.set_yticklabels([f"{th:.6f}" for th in thresholds])
        
        # Labels and title
        ax.set_xlabel('K Value (Homography Adaptation Iterations)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Detection Threshold', fontsize=12, fontweight='bold')
        ax.set_title('SuperPoint Keypoint Count: Threshold × K Parameter Space\n(Dataset: toy_33)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Average Number of Keypoints', fontsize=12, fontweight='bold')
        
        # Add value annotations
        for i in range(len(thresholds)):
            for j in range(len(k_values)):
                if not np.isnan(heatmap_data[i, j]):
                    text_color = 'white' if heatmap_data[i, j] < np.nanmean(heatmap_data) else 'black'
                    ax.text(j, i, f'{int(heatmap_data[i, j])}', 
                           ha='center', va='center', fontsize=9, fontweight='bold', color=text_color)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "keypoints_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run threshold × K sweep experiment')
    parser.add_argument('config', type=str, help='Path to base configuration file')
    parser.add_argument('output_dir', type=str, nargs='?', default=None, 
                       help='Output directory for experiments (default: auto-generated based on dataset)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=log_level
    )
    
    # Run experiment
    experiment = ThresholdKSweepExperiment(args.config, args.output_dir)
    experiment.run_experiments()


if __name__ == "__main__":
    main()