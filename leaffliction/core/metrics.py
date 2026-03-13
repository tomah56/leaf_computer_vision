"""Model evaluation and accuracy metrics."""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from .dataset import find_leaf_class_folders, create_custom_image_dataset
from .io_utils import get_cache_key, load_cached_results, save_cached_results


def evaluate_model_accuracy(model, data_dir: str, transform, batch_size: int = 16, model_path: str = None, use_cache: bool = True):
    """Evaluate model accuracy on a dataset and return metrics including confusion matrix."""
    # Check cache first
    if use_cache and model_path:
        cache_key = get_cache_key(model_path, data_dir)
        cached = load_cached_results(cache_key)
        if cached:
            return (
                cached['overall_accuracy'],
                cached['class_accuracies'],
                cached['class_names'],
                cached.get('confusion_matrix', None)
            )
    
    print("🔄 Evaluating model accuracy...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Find all leaf folders containing images
    leaf_folders = find_leaf_class_folders(data_dir)
    if not leaf_folders:
        raise ValueError("No image folders found in the dataset.")
    
    dataset = create_custom_image_dataset(leaf_folders, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    all_preds = []
    all_labels = []
    
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
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
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
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds).tolist()
    
    # Save to cache
    if use_cache and model_path:
        cache_key = get_cache_key(model_path, data_dir)
        save_cached_results(cache_key, overall_accuracy, class_accuracies, list(dataset.classes), conf_matrix)
    
    return overall_accuracy, class_accuracies, dataset.classes, conf_matrix
