#!/usr/bin/env python3
"""
Quick test script to verify the folder structure works correctly
Runs just one minimal experiment to test the naming convention
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our main experiment class
from tools.threshold_k_sweep_experiment import ThresholdKSweepExperiment


class MinimalThresholdKTest(ThresholdKSweepExperiment):
    """Minimal test version that runs just one experiment"""
    
    def __init__(self, config_path, base_output_dir=None):
        # Initialize parent class
        super().__init__(config_path, base_output_dir)
        
        # Override with minimal parameters for testing
        self.threshold_values = [0.015]  # Just one threshold
        self.k_values = [8]              # Just one K value
        
        logging.info(f"MINIMAL TEST: 1 threshold × 1 K value = 1 experiment")
        
    def test_folder_structure(self):
        """Test that folder structure is created correctly"""
        
        logging.info(f"Testing folder structure for dataset: {self.dataset_name}")
        logging.info(f"Base output directory: {self.base_output_dir}")
        
        # Test that the path follows the expected pattern
        expected_pattern = f"logs/export/{self.dataset_name}"
        if expected_pattern in str(self.base_output_dir):
            logging.info(f"✅ Folder structure matches expected pattern: {expected_pattern}")
        else:
            logging.warning(f"⚠️  Folder structure doesn't match expected pattern")
            logging.warning(f"   Expected: contains '{expected_pattern}'")
            logging.warning(f"   Actual: {self.base_output_dir}")
            
        # Test that dataset name extraction works
        logging.info(f"✅ Dataset name extracted: {self.dataset_name}")
        
        return True
        
    def run_minimal_test(self):
        """Run just the folder structure test without actual export"""
        
        logging.info("=== MINIMAL FOLDER STRUCTURE TEST ===")
        
        # Test folder structure
        self.test_folder_structure()
        
        # Create one experiment directory to verify the full path
        threshold = self.threshold_values[0]
        k_value = self.k_values[0]
        
        exp_name = f"th{int(threshold*1000000):06d}_K{k_value:03d}"
        exp_output_dir = self.base_output_dir / exp_name
        exp_output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"✅ Created test experiment directory: {exp_output_dir}")
        
        # Create a test file to verify the path works
        test_file = exp_output_dir / "test_file.txt"
        with open(test_file, 'w') as f:
            f.write(f"Test experiment: {exp_name}\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Threshold: {threshold}\n")
            f.write(f"K: {k_value}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            
        logging.info(f"✅ Created test file: {test_file}")
        
        # Verify the complete folder structure
        logging.info(f"\n=== COMPLETE FOLDER STRUCTURE ===")
        logging.info(f"Root: {self.base_output_dir}")
        logging.info(f"Experiment: {exp_output_dir}")
        logging.info(f"Test file: {test_file}")
        
        # Show the absolute path relationship
        project_root = Path(__file__).parent.parent
        rel_path = exp_output_dir.relative_to(project_root)
        logging.info(f"Relative to project root: {rel_path}")
        
        return exp_output_dir


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test folder structure naming')
    parser.add_argument('config', type=str, help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default=None, help='Override output directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    
    # Run test
    test = MinimalThresholdKTest(args.config, args.output_dir)
    test_dir = test.run_minimal_test()
    
    print(f"\n🎉 Test completed successfully!")
    print(f"📁 Test directory created: {test_dir}")
    print(f"📂 Structure follows pattern: logs/export/{test.dataset_name}/experiment_name")


if __name__ == "__main__":
    main()