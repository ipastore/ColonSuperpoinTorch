#!/bin/bash
# filepath: ~/storage/ColonSuperpoinTorch/dgx_scripts/run_export.sh

# Navigate to project directory
cd /workspace/ColonSuperpoinTorch

# Run the export command
python export.py export_detector_homoAdapt configs/superpoint_colon_export_test.yaml export_test
