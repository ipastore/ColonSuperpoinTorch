#!/usr/bin/env python3

import os
import sys
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
os.chdir(project_root)

from tools.th_k_sweep.threshold_k_sweep_experiment import ThresholdKSweepExperiment

def test_visualization():
    """Test the new visualization with existing results"""
    
    # Use existing results
    experiment = ThresholdKSweepExperiment(
        config_path="configs/superpoint_colon_export.yaml",
        base_output_dir="logs/export/toy_33/threshold_k_sweep_20250919_132915"
    )
    
    # Generate new plots with existing data
    experiment._generate_analysis_plots()
    print("✅ New visualization generated!")
    print("📁 Check: logs/export/toy_33/threshold_k_sweep_20250919_132915/analysis_plots/")

if __name__ == "__main__":
    test_visualization()