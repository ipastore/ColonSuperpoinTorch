#!/bin/bash

# Run camera/specular mask erosion sweep experiment
# Usage: ./run_mask_erosion_sweep.sh <config_path> [output_dir] [flags]

set -e

CONFIG_PATH="$1"

if [[ -z "$CONFIG_PATH" ]]; then
    echo "Usage: $0 <config_path> [output_dir] [flags]"
    echo ""
    echo "Arguments:"
    echo "  config_path           Path to SuperPoint configuration file"
    echo "  output_dir            Optional output directory for experiment results"
    echo ""
    echo "Flags:"
    echo "  --erode_camera N            Sweep erode_camera_mask 0..N"
    echo "  --erode_specular N          Sweep erode_specular_mask 0..N"
    echo "  --valid_border_margin N     Sweep valid_border_margin 0..N"
    echo "  --white_threshold X         Sweep specular_white_threshold up to X"
    echo "  --white_threshold_min X     Lower bound for white threshold sweep"
    echo "  --white_threshold_step X    Step size for white threshold sweep"
    echo "  --outputImg                 Save visualization overlays"
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

echo "Starting mask erosion sweep experiment..."
echo "Configuration: $CONFIG_PATH"
if [[ -n "$OUTPUT_DIR" ]]; then
    echo "Output directory: $OUTPUT_DIR"
    python tools/mask_erosion_sweep/mask_erosion_sweep_experiment.py "$CONFIG_PATH" "$OUTPUT_DIR" "${EXTRA_ARGS[@]}" --debug
else
    echo "Output directory: auto-generated based on dataset"
    python tools/mask_erosion_sweep/mask_erosion_sweep_experiment.py "$CONFIG_PATH" "${EXTRA_ARGS[@]}" --debug
fi

echo ""
echo "Sweep completed!"
if [[ -n "$OUTPUT_DIR" ]]; then
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "Check logs/export/<dataset>/mask_erosion_sweep_* for results."
fi
