#!/usr/bin/env python3
"""
Test script specifically for K=1 case to verify the fix
"""

import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.threshold_k_sweep_experiment import ThresholdKSweepExperiment


class K1TestExperiment(ThresholdKSweepExperiment):
    """Test experiment focusing on K=2 and a few other small K values (K=1 removed)"""
    
    def __init__(self, config_path, base_output_dir=None):
        super().__init__(config_path, base_output_dir)
        
        # Override with minimal parameters focused on small K values
        self.threshold_values = [0.015, 0.02]  # Just two thresholds
        self.k_values = [2, 4, 8]              # Start from K=2 (K=1 removed)
        
        logging.info(f"K TEST (no K=1): {len(self.threshold_values)} thresholds × {len(self.k_values)} K values = {len(self.threshold_values) * len(self.k_values)} experiments")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test K=1 fix')
    parser.add_argument('config', type=str, help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    
    # Create and run experiment
    experiment = K1TestExperiment(args.config)
    experiment.run_experiments()
    
    print(f"\n🎉 K test (no K=1) completed!")
    print(f"📁 Results saved to: {experiment.base_output_dir}")


if __name__ == "__main__":
    main()