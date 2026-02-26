from pathlib import Path
import json
import hashlib
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, datasets
from torchvision.models import ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from transformation import get_transforms, MEAN, STD

CACHE_FILE = "accuracy_cache.json"

def load_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(class_to_idx)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, class_to_idx

def predict_image(image_path: str, model, class_to_idx: dict):
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    transform = get_transforms(train=False)

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(x)
        pred_idx = outputs.argmax(dim=1).item()

    return idx_to_class[pred_idx]


def visualize_prediction(image_path: str, model, class_to_idx: dict):
    transform = get_transforms(train=False)
    image = Image.open(image_path).convert("RGB")
    x = transform(image)

    with torch.no_grad():
        outputs = model(x.unsqueeze(0))
        pred_idx = outputs.argmax(dim=1).item()

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    label = idx_to_class[pred_idx]

    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    x_vis = x * std + mean
    x_vis = torch.clamp(x_vis, 0, 1)
    x_vis = x_vis.permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(x_vis)
    axes[1].set_title("Transformed")
    axes[1].axis("off")

    fig.text(0.5, 0.02, f"Prediction: {label}", ha="center", fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
    return label


def get_cache_key(model_path: str, data_dir: str) -> str:
    """Generate a unique cache key based on model and data directory."""
    # Get modification times
    model_mtime = os.path.getmtime(model_path)
    data_mtime = max(
        os.path.getmtime(os.path.join(root, f))
        for root, _, files in os.walk(data_dir)
        for f in files
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ) if os.path.exists(data_dir) else 0
    
    # Create hash from paths and modification times
    cache_str = f"{model_path}:{model_mtime}:{data_dir}:{data_mtime}"
    return hashlib.md5(cache_str.encode()).hexdigest()


def load_cached_results(cache_key: str):
    """Load cached accuracy results if they exist."""
    if not os.path.exists(CACHE_FILE):
        return None
    
    try:
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        
        if cache_key in cache:
            print("📦 Loading cached accuracy results...")
            return cache[cache_key]
    except (json.JSONDecodeError, KeyError):
        pass
    
    return None


def save_cached_results(cache_key: str, overall_acc: float, class_accs: dict, class_names: list):
    """Save accuracy results to cache."""
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)
        except json.JSONDecodeError:
            pass
    
    cache[cache_key] = {
        'overall_accuracy': overall_acc,
        'class_accuracies': class_accs,
        'class_names': class_names
    }
    
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)
    
    print("💾 Results cached for future use")


def evaluate_model_accuracy(model, data_dir: str, batch_size: int = 16, model_path: str = None, use_cache: bool = True):
    """Evaluate model accuracy on a dataset and return metrics."""
    # Check cache first
    if use_cache and model_path:
        cache_key = get_cache_key(model_path, data_dir)
        cached = load_cached_results(cache_key)
        if cached:
            return (
                cached['overall_accuracy'],
                cached['class_accuracies'],
                cached['class_names']
            )
    
    print("🔄 Evaluating model accuracy...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    transform = get_transforms(train=False)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    
    # Initialize counters for each class
    for class_name in dataset.classes:
        class_correct[class_name] = 0
        class_total[class_name] = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Track per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = preds[i].item()
                class_name = dataset.classes[label]
                class_total[class_name] += 1
                if label == pred:
                    class_correct[class_name] += 1
    
    overall_accuracy = correct / total if total > 0 else 0
    class_accuracies = {
        cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
        for cls in dataset.classes
    }
    
    # Save to cache
    if use_cache and model_path:
        cache_key = get_cache_key(model_path, data_dir)
        save_cached_results(cache_key, overall_accuracy, class_accuracies, list(dataset.classes))
    
    return overall_accuracy, class_accuracies, dataset.classes


def visualize_accuracy(model, data_dir: str, batch_size: int = 16, model_path: str = None, use_cache: bool = True):
    """Create a simple visual representation of model accuracy."""
    overall_acc, class_accs, class_names = evaluate_model_accuracy(
        model, data_dir, batch_size, model_path, use_cache
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Overall accuracy gauge
    ax1 = axes[0]
    ax1.barh([0], [overall_acc * 100], color='#4CAF50', height=0.5)
    ax1.barh([0], [100 - overall_acc * 100], left=[overall_acc * 100], 
             color='#E0E0E0', height=0.5)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'Overall Model Accuracy: {overall_acc * 100:.2f}%', 
                  fontsize=14, fontweight='bold')
    ax1.set_yticks([])
    ax1.grid(axis='x', alpha=0.3)
    
    # Add percentage text
    ax1.text(overall_acc * 50, 0, f'{overall_acc * 100:.1f}%', 
            ha='center', va='center', fontsize=16, fontweight='bold', color='white')
    
    # Right plot: Per-class accuracy bars
    ax2 = axes[1]
    y_pos = np.arange(len(class_names))
    accuracies = [class_accs[cls] * 100 for cls in class_names]
    
    colors = ['#4CAF50' if acc >= 80 else '#FFC107' if acc >= 60 else '#F44336' 
              for acc in accuracies]
    bars = ax2.barh(y_pos, accuracies, color=colors)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([cls.replace('_', ' ') for cls in class_names])
    ax2.set_xlabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add percentage labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax2.text(acc + 2, i, f'{acc:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*50)
    print(f"MODEL ACCURACY REPORT")
    print("="*50)
    print(f"Overall Accuracy: {overall_acc * 100:.2f}%")
    print("\nPer-Class Accuracy:")
    for cls in class_names:
        print(f"  {cls.replace('_', ' ')}: {class_accs[cls] * 100:.2f}%")
    print("="*50 + "\n")


if __name__ == "__main__":
    model, class_to_idx = load_model("model_split.pth")
    
    # Visualize model accuracy
    print("Evaluating model accuracy...")
    visualize_accuracy(model, "images/apple")
    
    # Individual predictions
    visualize_prediction("images/apple/Apple_scab/image (10).JPG", model, class_to_idx)
    # visualize_prediction("leaffliction/images/apple/Apple_scab/image (10).JPG", model, class_to_idx)
    # visualize_prediction("images/apple/Apple_rust/image (13).JPG", model, class_to_idx)
    # visualize_prediction("images/apple/Apple_Black_rot/image (37).JPG", model, class_to_idx)
    # visualize_prediction("images/apple/Apple_healthy/image (39).JPG", model, class_to_idx)
