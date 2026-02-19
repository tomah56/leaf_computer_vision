#!/usr/bin/env python3
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


data_dir = Path("images/apple")
epochs = 2
batch_size = 16
learning_rate = 1e-3
num_workers = 2
log_every = 10
val_split = 0.2  # 20% of data for validation


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

	print(f"Found {len(dataset)} images across {len(dataset.classes)} classes: {dataset.classes}")

	# Split dataset into training and validation
	val_size = int(len(dataset) * val_split)
	train_size = len(dataset) - val_size
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
	
	print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
	)
	
	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
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
		# Training phase
		model.train()
		running_loss = 0.0
		running_correct = 0
		total = 0

		for batch_idx, (images, labels) in enumerate(train_loader, start=1):
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

			if batch_idx % log_every == 0:
				print(
					f"Epoch {epoch + 1}/{epochs} - batch {batch_idx}/{len(train_loader)} - "
					f"loss: {loss.item():.4f}"
				)

		train_loss = running_loss / max(total, 1)
		train_acc = running_correct / max(total, 1)
		print(f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f}")

		# Validation phase
		model.eval()
		val_running_loss = 0.0
		val_running_correct = 0
		val_total = 0

		with torch.no_grad():
			for images, labels in val_loader:
				images = images.to(device)
				labels = labels.to(device)

				outputs = model(images)
				loss = criterion(outputs, labels)

				val_running_loss += loss.item() * images.size(0)
				_, preds = torch.max(outputs, 1)
				val_running_correct += (preds == labels).sum().item()
				val_total += images.size(0)

		val_loss = val_running_loss / max(val_total, 1)
		val_acc = val_running_correct / max(val_total, 1)
		print(f"Epoch {epoch + 1}/{epochs} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

	checkpoint = {
		"model_state": model.state_dict(),
		"class_to_idx": dataset.class_to_idx,
		"arch": "resnet18",
	}
	torch.save(checkpoint, "model_split.pth")

if __name__ == "__main__":
	main()
