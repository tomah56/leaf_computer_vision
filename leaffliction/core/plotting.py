"""Visualization utilities for predictions and accuracy metrics."""

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

from .metrics import evaluate_model_accuracy


def visualize_prediction(image_path: str, model, class_to_idx: dict, transform, mean, std):
    """Visualize a prediction with original and transformed images."""
    image = Image.open(image_path).convert("RGB")
    x = transform(image)

    with torch.no_grad():
        outputs = model(x.unsqueeze(0))
        pred_idx = outputs.argmax(dim=1).item()

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    label = idx_to_class[pred_idx]

    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    x_vis = x * std_tensor + mean_tensor
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


def visualize_accuracy(model, data_dir: str, transform, batch_size: int = 16, model_path: str = None, use_cache: bool = True):
    """Create a visual representation of model accuracy with confusion matrix."""
    overall_acc, class_accs, class_names, conf_matrix = evaluate_model_accuracy(
        model, data_dir, transform, batch_size, model_path, use_cache
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Top-left: Overall accuracy gauge
    ax1 = axes[0, 0]
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
    
    # Top-right: Per-class accuracy bars
    ax2 = axes[0, 1]
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
    
    # Bottom: Confusion matrix heatmap
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    if conf_matrix is not None:
        conf_matrix_array = np.array(conf_matrix)
        sns.heatmap(conf_matrix_array, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[cls.replace('_', ' ') for cls in class_names],
                    yticklabels=[cls.replace('_', ' ') for cls in class_names],
                    ax=ax3, cbar=True)
        ax3.set_xlabel('Predicted', fontsize=12)
        ax3.set_ylabel('True', fontsize=12)
        ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'Confusion matrix unavailable\n(cached results)', 
                ha='center', va='center', fontsize=12, transform=ax3.transAxes)
        ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax3.axis('off')
    
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
