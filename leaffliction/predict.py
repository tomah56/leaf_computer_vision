#!/usr/bin/env python3
"""Prediction script for leaf disease classification."""

import json
import argparse

from core.model import load_model
from core.plotting import visualize_prediction, visualize_accuracy
from transformation import get_transforms, MEAN, STD


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model predictions on an image or based on a JSON configuration file")
    parser.add_argument("input", type=str, help="Path to JSON configuration file or image file (JPG/PNG)")
    args = parser.parse_args()
    
    input_path = args.input
    
    # Determine if input is a JSON config or an image file
    if input_path.lower().endswith('.json'):
        # Load configuration
        with open(input_path, 'r') as f:
            config = json.load(f)
        
        model_path = config["model"]
        folder = config["folder"]
        images = config["images"]
        
        # Load model
        print(f"Loading model: {model_path}")
        model, class_to_idx = load_model(model_path)
        
        # Visualize model accuracy (with caching)
        print(f"\nEvaluating model on: {folder}")
        transform = get_transforms(train=False)
        visualize_accuracy(model, folder, transform, model_path=model_path)
        
        # Individual predictions
        print(f"\nRunning predictions on {len(images)} images...")
        for image_path in images:
            print(f"\nPredicting: {image_path}")
            visualize_prediction(image_path, model, class_to_idx, transform, MEAN, STD)
    
    elif input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
        # Single image prediction with default model
        default_model = "model.pth"
        print(f"Loading default model: {default_model}")
        model, class_to_idx = load_model(default_model)
        
        print(f"\nPredicting: {input_path}")
        transform = get_transforms(train=False)
        visualize_prediction(input_path, model, class_to_idx, transform, MEAN, STD)
    
    else:
        print(f"Error: Input must be a JSON file (.json) or an image file (.jpg, .jpeg, .png)")
        exit(1)