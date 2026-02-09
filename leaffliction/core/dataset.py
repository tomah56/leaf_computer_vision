import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class LeafDataset(Dataset):
    def __init__(self, dataset_dict, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        # map folder names to indices
        for idx, folder in enumerate(dataset_dict.keys()):
            self.class_to_idx[os.path.basename(folder)] = idx
            for file in dataset_dict[folder]:
                self.samples.append((os.path.join(folder, file), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
