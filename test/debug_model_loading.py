#!/usr/bin/env python
"""
Debug script to identify which model is being loaded with different configuration names.

This script helps diagnose model loading issues by printing detailed information
about the loaded model class, including its module path, structure, and parameter counts.
"""

import argparse
import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, Type

import torch
import yaml


# Add parent directory to path so we can import modules from the main package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.loader import get_model, modelLoader


def print_model_info(model_class: Type[Any], model_name: str) -> None:
    """Print detailed information about a model class.
    
    Args:
        model_class: The model class to inspect
        model_name: The name of the model used in configuration
        
    Returns:
        None, prints information to console
    """
    print(f"\n{'='*80}")
    print(f"Model name: {model_name}")
    print(f"Model class: {model_class.__name__}")
    print(f"Defined in module: {model_class.__module__}")
    print(f"File location: {inspect.getfile(model_class)}")
    
    # Check if it has batch normalization
    model_source = inspect.getsource(model_class)
    has_batchnorm = "BatchNorm" in model_source
    print(f"Contains BatchNorm layers: {has_batchnorm}")
    
    # Instantiate model to check parameters
    try:
        model = model_class()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Get parameter names (useful to check if they match expected format)
        print("\nFirst 10 parameter names:")
        for i, (name, _) in enumerate(model.named_parameters()):
            if i >= 10:
                break
            print(f"  {name}")
        
    except Exception as e:
        print(f"Error instantiating model: {e}")
    
    print(f"{'='*80}\n")


def main() -> None:
    """Main function to debug model loading."""
    parser = argparse.ArgumentParser(description="Debug model loading in ColonSuperpoinTorch")
    parser.add_argument("--config", type=str, default=None, 
                      help="Path to config YAML file")
    parser.add_argument("--model_name", type=str, default=None,
                      help="Model name to test (overrides config)")
    args = parser.parse_args()
    
    # Model names to test if not specified
    model_names = ["SuperPointNet_pretrained", "SuperPointNet", "SuperPointNet_gauss2"]
    
    if args.config:
        # Load from config file
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        if args.model_name:
            model_names = [args.model_name]
        elif 'model' in config and 'name' in config['model']:
            model_names = [config['model']['name']]
    elif args.model_name:
        model_names = [args.model_name]
    
    print(f"Testing model loading for: {', '.join(model_names)}")
    
    # First try direct class loading
    for name in model_names:
        try:
            print(f"\nTesting direct model class loading for: {name}")
            model_class = get_model(name)
            print_model_info(model_class, name)
        except Exception as e:
            print(f"Error loading model class {name}: {e}")
    
    # Then try with modelLoader which is used in training
    for name in model_names:
        try:
            print(f"\nTesting modelLoader for: {name}")
            model = modelLoader(model=name)
            print(f"ModelLoader created instance of: {type(model).__name__}")
            print(f"From module: {type(model).__module__}")
        except Exception as e:
            print(f"Error with modelLoader for {name}: {e}")


if __name__ == "__main__":
    main()