from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

def load_model(checkpoint_path: str, num_classes: int):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["class_to_idx"]

def predict_image(image_path: str, model, class_to_idx: dict):
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(x)
        pred_idx = outputs.argmax(dim=1).item()

    return idx_to_class[pred_idx]

if __name__ == "__main__":
    model, class_to_idx = load_model("model.pth", num_classes=1 + max(class_to_idx.values()))
    # Replace with your test image path:
    label = predict_image("images/apple/Apple_scab/your_test.jpg", model, class_to_idx)
    print("Predicted class:", label)