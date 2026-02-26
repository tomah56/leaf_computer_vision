#!/usr/bin/env python3
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models
from torchvision.models import ResNet18_Weights

from transformation import get_transforms


data_dir = Path("images/apple")
epochs = 2
batch_size = 16
learning_rate = 1e-3
num_workers = 2
log_every = 10
val_split = 0.2  # 20% of data for validation
seed = 42


def stratified_split(targets, split_ratio, rng_seed):
	class_indices = {}
	for idx, label in enumerate(targets):
		class_indices.setdefault(label, []).append(idx)

	generator = torch.Generator().manual_seed(rng_seed)
	train_indices = []
	val_indices = []

	for _, indices in class_indices.items():
		if len(indices) <= 1:
			train_indices.extend(indices)
			continue

		perm = torch.randperm(len(indices), generator=generator).tolist()
		shuffled = [indices[i] for i in perm]
		raw_val_count = int(len(indices) * split_ratio)
		val_count = max(1, min(len(indices) - 1, raw_val_count))
		val_indices.extend(shuffled[:val_count])
		train_indices.extend(shuffled[val_count:])

	return train_indices, val_indices


def main():
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	transform = get_transforms(train=True)

	dataset = datasets.ImageFolder(root=str(data_dir), transform=transform)
	if len(dataset.classes) == 0:
		raise ValueError("No classes found in the dataset.")

	print(
		f"Found {len(dataset)} images across {len(dataset.classes)} "
		f"classes: {dataset.classes}"
	)

	train_indices, val_indices = stratified_split(
		dataset.targets, val_split, seed
	)
	train_dataset = Subset(dataset, train_indices)
	val_dataset = Subset(dataset, val_indices)

	print(
		f"Training samples: {len(train_dataset)}, "
		f"Validation samples: {len(val_dataset)}"
	)

	class_counts = torch.zeros(len(dataset.classes), dtype=torch.float)
	for idx in train_indices:
		class_counts[dataset.targets[idx]] += 1

	class_weights = class_counts.sum() / (class_counts * len(dataset.classes))
	class_weights = torch.where(
		class_counts > 0, class_weights, torch.zeros_like(class_weights)
	)

	sample_weights = [
		class_weights[dataset.targets[idx]].item() for idx in train_indices
	]
	sampler = WeightedRandomSampler(
		sample_weights, num_samples=len(sample_weights), replacement=True
	)

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		sampler=sampler,
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

	criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

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
					f"Epoch {epoch + 1}/{epochs} - batch "
					f"{batch_idx}/{len(train_loader)} - "
					f"loss: {loss.item():.4f}"
				)

		train_loss = running_loss / max(total, 1)
		train_acc = running_correct / max(total, 1)
		print(
			f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f} - "
			f"train_acc: {train_acc:.4f}"
		)

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
		print(
			f"Epoch {epoch + 1}/{epochs} - val_loss: {val_loss:.4f} - "
			f"val_acc: {val_acc:.4f}"
		)

		scheduler.step()

	checkpoint = {
		"model_state": model.state_dict(),
		"class_to_idx": dataset.class_to_idx,
		"arch": "resnet18",
	}
	torch.save(checkpoint, "model_gaussian.pth")

if __name__ == "__main__":
	main()
