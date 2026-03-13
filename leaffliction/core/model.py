"""Model loading and prediction utilities."""

import torch
from torch import nn
from torchvision import models
from PIL import Image


def load_model(checkpoint_path: str):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(class_to_idx)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, class_to_idx


def predict_image(image_path: str, model, class_to_idx: dict, transform):
    """Predict the class of a single image."""
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(x)
        pred_idx = outputs.argmax(dim=1).item()

    return idx_to_class[pred_idx]
