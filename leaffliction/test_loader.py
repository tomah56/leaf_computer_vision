# test_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from core.dataset import LeafDataset
from distribution import load_dataset  # reuse your existing loader

# -----------------------------
# 1. Load dataset dictionary
# -----------------------------
dataset_dict, _ = load_dataset("./images")  # adjust path if needed


# -----------------------------
# 2. Create LeafDataset
# -----------------------------
leaf_dataset = LeafDataset(dataset_dict)

# -----------------------------
# 3. Wrap in DataLoader
# -----------------------------
loader = DataLoader(leaf_dataset, batch_size=4, shuffle=True)

# -----------------------------
# 4. Test one batch
# -----------------------------
images, labels = next(iter(loader))

print(f"Images batch shape: {images.shape}")  # should be [batch_size, 3, 224, 224]
print(f"Labels batch: {labels}")              # should be integers

# -----------------------------
# 5. Optional: check class names
# -----------------------------
print(f"Class mapping: {leaf_dataset.idx_to_class}")
