from torchvision import transforms

IMAGE_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool = False):
	transform_steps = [transforms.Resize(IMAGE_SIZE)]

	if train:
		transform_steps.append(transforms.RandomHorizontalFlip())

	transform_steps.extend(
		[
			transforms.ToTensor(),
			transforms.Normalize(mean=MEAN, std=STD),
		]
	)

	return transforms.Compose(transform_steps)
