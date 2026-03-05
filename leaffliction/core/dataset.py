"""
LeafDataset - PyTorch-compatible dataset loader for leaf disease classification

- Returns individual image tensors and numeric class labels
- Handles full image paths
- Accepts transformations/augmentations for training
- Compatible with DataLoader for batching and shuffling
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


def find_leaf_class_folders(data_dir):
    """Find all leaf folders containing images and return a mapping."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    leaf_folders = []
    
    for root, dirs, files in os.walk(str(data_dir)):
        # Check if this folder contains image files
        has_images = any(
            Path(f).suffix.lower() in image_extensions 
            for f in files
        )
        if has_images:
            leaf_folders.append(Path(root))
    
    return sorted(leaf_folders)


def create_custom_image_dataset(leaf_folders, transform):
    """Create a custom dataset from leaf folders."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    
    # Map folder paths to class indices
    class_to_idx = {folder.name: idx for idx, folder in enumerate(leaf_folders)}
    
    samples = []
    for class_idx, folder in enumerate(leaf_folders):
        for file in sorted(folder.iterdir()):
            if file.suffix.lower() in image_extensions:
                samples.append((str(file), class_idx))
    
    class ImageDataset(Dataset):
        def __init__(self, samples, class_to_idx, transform=None):
            self.samples = samples
            self.class_to_idx = class_to_idx
            self.transform = transform
            self.classes = sorted(class_to_idx.keys())
            self.targets = [s[1] for s in samples]
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            path, label = self.samples[idx]
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
    
    return ImageDataset(samples, class_to_idx, transform)

class LeafDataset(Dataset):
    def __init__(self, dataset_dict, transform=None):
        """
        Args:
            dataset_dict (dict): {folder_path: [list of image filenames]}
            transform (callable, optional): torchvision transform to apply
        """
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # map folder names to indices
        for idx, folder in enumerate(dataset_dict.keys()):
            class_name = os.path.basename(folder)
            self.class_to_idx[class_name] = idx
            for file in dataset_dict[folder]:
                full_path = os.path.abspath(os.path.join(folder, file))
                self.samples.append((full_path, idx))

        # inverse mapping for predictions
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {path}: {e}")

        if self.transform:
            img = self.transform(img)
        return img, label

    def __repr__(self):
        return f"<LeafDataset len={len(self)} classes={list(self.class_to_idx.keys())}>"
