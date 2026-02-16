from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


data_dir = Path("images/apple")
epochs = 2
batch_size = 16
learning_rate = 1e-3
num_workers = 2


def main():
	transform = transforms.Compose(
		[
			transforms.Resize((224, 224)),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)

	dataset = datasets.ImageFolder(root=str(data_dir), transform=transform)
	if len(dataset.classes) == 0:
		raise ValueError("No classes found in the dataset.")

	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
	model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
	model = model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	for epoch in range(epochs):
		model.train()
		running_loss = 0.0
		running_correct = 0
		total = 0

		for images, labels in loader:
			images = images.to(device)
			labels = labels.to(device)

			optimizer.zero_grad(set_to_none=True)
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item() * images.size(0)
			_, preds = torch.max(outputs, 1)
			running_correct += (preds == labels).sum().item()
			total += images.size(0)

		avg_loss = running_loss / max(total, 1)
		avg_acc = running_correct / max(total, 1)
		print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} - acc: {avg_acc:.4f}")

	checkpoint = {
		"model_state": model.state_dict(),
		"class_to_idx": dataset.class_to_idx,
		"arch": "resnet18",
	}
	torch.save(checkpoint, "model.pth")

if __name__ == "__main__":
	main()
