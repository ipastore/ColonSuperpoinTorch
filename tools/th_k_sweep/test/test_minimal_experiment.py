#!/usr/bin/env python3
"""
Run a minimal actual experiment with 2 threshold × 2 K combinations
to verify the complete pipeline works with the new folder structure
"""

import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.threshold_k_sweep_experiment import ThresholdKSweepExperiment


class MinimalExperiment(ThresholdKSweepExperiment):
    """Minimal experiment with just 4 total combinations"""
    
    def __init__(self, config_path, base_output_dir=None):
        super().__init__(config_path, base_output_dir)
        
        # Override with minimal parameters for testing (2x2 = 4 experiments)
        self.threshold_values = [0.010, 0.020]  # Just two thresholds
        self.k_values = [4, 8]                  # Just two K values
        
        logging.info(f"MINIMAL EXPERIMENT: {len(self.threshold_values)} thresholds × {len(self.k_values)} K values = {len(self.threshold_values) * len(self.k_values)} experiments")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run minimal experiment to test folder structure')
    parser.add_argument('config', type=str, help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    
    # Create and run experiment
    experiment = MinimalExperiment(args.config)
    experiment.run_experiments()
    
    print(f"\n🎉 Minimal experiment completed!")
    print(f"📁 Results saved to: {experiment.base_output_dir}")
    print(f"📂 Folder structure: logs/export/{experiment.dataset_name}/threshold_k_sweep_TIMESTAMP/")


if __name__ == "__main__":
    main()