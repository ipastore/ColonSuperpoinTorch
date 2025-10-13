#!/bin/bash
# filepath: ~/storage/ColonSuperpoinTorch/dgx_scripts/run_joint_train.sh

# Navigate to project directory
cd /workspace/ColonSuperpoinTorch

# Run the export command
python train4.py joint_train configs/superpoint_colon_train_heatmap.yaml train_test --eval
