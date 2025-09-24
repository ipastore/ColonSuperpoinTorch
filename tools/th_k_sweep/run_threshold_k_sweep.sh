#!/bin/bash

# Run Threshold × K Sweep Experiment
# Usage: ./run_threshold_k_sweep.sh <config_path> [output_dir]

set -e

# Parse arguments
CONFIG_PATH="$1"
OUTPUT_DIR="$2"  # Optional - will be auto-generated if not provided

# Validate arguments
if [[ -z "$CONFIG_PATH" ]]; then
    echo "Usage: $0 <config_path> [output_dir]"
    echo ""
    echo "Arguments:"
    echo "  config_path  Path to SuperPoint configuration file (e.g., configs/superpoint_colon_export.yaml)"
    echo "  output_dir   Output directory for experiment results (optional - auto-generated based on dataset and timestamp)"
    echo ""
    echo "Example:"
    echo "  $0 configs/superpoint_colon_export.yaml"
    echo "  $0 configs/superpoint_colon_export.yaml logs/export/toy_33/my_custom_experiment"
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: Configuration file '$CONFIG_PATH' not found!"
    exit 1
fi

# Activate conda environment
echo "Activating conda environment py38-sp..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate py38-sp

# Confirm Python version
echo "Python version: $(python --version)"

# Run the experiment
echo "Starting Threshold × K sweep experiment..."
echo "Configuration: $CONFIG_PATH"
if [[ -n "$OUTPUT_DIR" ]]; then
    echo "Output directory: $OUTPUT_DIR"
    python tools/th_k_sweep/threshold_k_sweep_experiment.py "$CONFIG_PATH" "$OUTPUT_DIR" --debug
else
    echo "Output directory: auto-generated based on dataset"
    python tools/th_k_sweep/threshold_k_sweep_experiment.py "$CONFIG_PATH" --debug
fi

# Summary
OUTPUT_DIR_FOR_SUMMARY="${OUTPUT_DIR:-logs/export/*/threshold_k_sweep_*}"
echo ""
echo "Experiment completed!"
echo "Results saved to: $OUTPUT_DIR_FOR_SUMMARY"
echo ""
echo "Generated files:"
echo "  - threshold_k_sweep_results.json    (detailed results)"
echo "  - threshold_k_sweep_results.csv     (tabular format)"
echo "  - experiment_summary.txt            (summary statistics)"
echo "  - analysis_plots/                   (visualization plots)"
echo ""
if [[ -n "$OUTPUT_DIR" ]]; then
    echo "To view visualizations:"
    echo "  ls $OUTPUT_DIR/analysis_plots/"
    echo ""
    echo "To analyze results:"
    echo "  cat $OUTPUT_DIR/experiment_summary.txt"
else
    echo "To find your experiment results:"
    echo "  ls -la logs/export/*/threshold_k_sweep_*/"
fi