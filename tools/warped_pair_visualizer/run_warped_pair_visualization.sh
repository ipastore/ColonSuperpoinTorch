#!/bin/bash

# Visualize warped pairs using a training configuration
# Usage: ./run_warped_pair_visualization.sh <config_path> [output_dir] [flags]

set -e

CONFIG_PATH="$1"

if [[ -z "$CONFIG_PATH" ]]; then
    echo "Usage: $0 <config_path> [output_dir] [flags]"
    echo ""
    echo "Arguments:"
    echo "  config_path  Path to training config YAML"
    echo "  output_dir   Optional directory for visualizations"
    echo ""
    echo "Flags:"
    echo "  --split {train|val}      Dataset split to visualize"
    echo "  --num-samples N          Number of samples to export"
    echo "  --debug                  Enable debug logging"
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: Configuration file '$CONFIG_PATH' not found!"
    exit 1
fi

shift

OUTPUT_DIR=""
if [[ $# -gt 0 && "$1" != --* ]]; then
    OUTPUT_DIR="$1"
    shift
fi

EXTRA_ARGS=("$@")

echo "Activating conda environment py38-sp..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate py38-sp

echo "Python version: $(python --version)"

echo "Starting warped pair visualization..."
echo "Configuration: $CONFIG_PATH"
if [[ -n "$OUTPUT_DIR" ]]; then
    echo "Output directory: $OUTPUT_DIR"
    python tools/warped_pair_visualizer/warped_pair_visualization.py "$CONFIG_PATH" "$OUTPUT_DIR" "${EXTRA_ARGS[@]}"
else
    echo "Output directory: auto-generated"
    python tools/warped_pair_visualizer/warped_pair_visualization.py "$CONFIG_PATH" "${EXTRA_ARGS[@]}"
fi

echo "Visualization complete."
