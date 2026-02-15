import sys
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
    print("Augmentations saved in:")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python Augmentation.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    augment_image(image_path)
