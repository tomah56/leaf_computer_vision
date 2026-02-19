from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt

def load_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(class_to_idx)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, class_to_idx

def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict_image(image_path: str, model, class_to_idx: dict):
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    transform = build_transform()

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(x)
        pred_idx = outputs.argmax(dim=1).item()

    return idx_to_class[pred_idx]


def visualize_prediction(image_path: str, model, class_to_idx: dict):
    transform = build_transform()
    image = Image.open(image_path).convert("RGB")
    x = transform(image)

    with torch.no_grad():
        outputs = model(x.unsqueeze(0))
        pred_idx = outputs.argmax(dim=1).item()

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    label = idx_to_class[pred_idx]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
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

if __name__ == "__main__":
    model, class_to_idx = load_model("model_split.pth")
    # Replace with your test image path:
    visualize_prediction("images/apple/Apple_rust/image (13).JPG", model, class_to_idx)
    visualize_prediction("images/apple/Apple_Black_rot/image (37).JPG", model, class_to_idx)
    visualize_prediction("images/apple/Apple_healthy/image (39).JPG", model, class_to_idx)