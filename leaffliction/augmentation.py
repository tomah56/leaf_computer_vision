#!/usr/bin/env python3

import os
import sys
from PIL import Image
import torchvision.transforms as transforms


augmentations = {
    "Flip": transforms.RandomHorizontalFlip(p=0.5),

    "Rotate": transforms.RandomRotation(degrees=25),

    "Skew": transforms.RandomAffine(
        degrees=0,
        shear=(-10, 10)
    ),

    "Shear": transforms.RandomAffine(
        degrees=0,
        shear=(-5, 5)
    ),

    "Crop": transforms.RandomResizedCrop(
        size=(224, 224),
        scale=(0.8, 1.0)
    ),

    "Distortion": transforms.ElasticTransform(alpha=10.0)
}


def augment_image(image_path):

    image = Image.open(image_path).convert("RGB")

    relative_path = os.path.normpath(image_path)
    parts = relative_path.split(os.sep)

    folders = parts[:-1]
    filename = parts[-1]
    name, ext = os.path.splitext(filename)

    save_root = os.path.join("images/augmented_directory", *folders)
    os.makedirs(save_root, exist_ok=True)

    image.save(os.path.join(save_root, filename))

    for aug_name, transform in augmentations.items():
        augmented_img = transform(image)
        new_filename = f"{name}_{aug_name}{ext}"
        print("new file: " + new_filename)
        augmented_img.save(os.path.join(save_root, new_filename))

    print("Augmentations saved in:", save_root)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python Augmentation.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    augment_image(image_path)
