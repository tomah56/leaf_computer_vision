from torchvision import transforms

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool = False):
	if train:
		transform_steps = [
			transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomRotation(degrees=15),
			transforms.ColorJitter(
				brightness=0.2,
				contrast=0.2,
				saturation=0.2,
				hue=0.05,
			),
			transforms.RandomApply(
				[transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
				p=0.2,
			),
		]
	else:
		transform_steps = [
			transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
			transforms.CenterCrop(IMAGE_SIZE),
		]

	transform_steps.extend(
		[
			transforms.ToTensor(),
			transforms.Normalize(mean=MEAN, std=STD),
		]
	)

	return transforms.Compose(transform_steps)
