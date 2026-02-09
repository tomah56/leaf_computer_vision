"""
LeafDataset - PyTorch-compatible dataset loader for leaf disease classification

- Returns individual image tensors and numeric class labels
- Handles full image paths
- Accepts transformations/augmentations for training
- Compatible with DataLoader for batching and shuffling
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset

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
