#!/usr/bin/env python3
"""
balance_dataset.py

Given an input folder of categorised leaf images, this script:

  1. Reads every sub-folder that contains images (leaf of the tree).
  2. Randomly picks TEST_PER_CLASS (10) images per category and copies
     them to   <parent>/test_images/<relative_category_path>/
  3. Copies the remaining originals to
               <parent>/images_modified/<relative_category_path>/
  4. If a category has fewer than MAX_IMAGES (600) training images,
     generates augmented copies by cycling through augmentations
     until the folder reaches exactly MAX_IMAGES files.
     Categories that already have >= MAX_IMAGES are left as-is.

Usage:
    python balance_dataset.py <input_images_dir>

Example:
    python balance_dataset.py leaffliction/images
"""

import os
import sys
import random
import shutil

from PIL import Image
from augmentation import augmentations


# ── constants ────────────────────────────────────────────────────────────────

MAX_IMAGES = 600
TEST_PER_CLASS = 10

IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png',
    '.gif', '.bmp', '.tiff', '.webp',
}

# Augmentation pool – built from the shared augmentations dict in augmentation.py.
AUGMENTATIONS = list(augmentations.values())


# ── helpers ───────────────────────────────────────────────────────────────────


def is_valid_image(filepath):
    """
    Return True if *filepath* is a readable image file.

    Args:
        filepath (str): absolute path to a file.

    Returns:
        bool: True when the file can be opened and loaded as an image.
    """
    try:
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            img.load()
        return True
    except Exception:
        return False


def collect_categories(root_dir):
    """
    Walk *root_dir* and return one entry per leaf directory that
    contains at least one valid image.

    Args:
        root_dir (str): absolute path to the top-level image directory.

    Returns:
        dict[str, list[str]]:
            mapping  relative_category_path -> [absolute_image_paths]
    """
    categories = {}

    for dirpath, _dirs, filenames in os.walk(root_dir):
        images = []
        for fname in filenames:
            _, ext = os.path.splitext(fname)
            if ext.lower() not in IMAGE_EXTENSIONS:
                continue
            full = os.path.join(dirpath, fname)
            if is_valid_image(full):
                images.append(full)

        if images:
            rel = os.path.relpath(dirpath, root_dir)
            categories[rel] = images

    return categories


def apply_augmentation(image, aug_index):
    """
    Apply one augmentation from AUGMENTATIONS to *image*.

    The transform is chosen by ``aug_index % len(AUGMENTATIONS)``,
    so cycling the index produces a varied sequence of transforms.

    Args:
        image (PIL.Image.Image): source image (RGB).
        aug_index (int): determines which augmentation to apply.

    Returns:
        PIL.Image.Image: augmented image (RGB).
    """
    aug = AUGMENTATIONS[aug_index % len(AUGMENTATIONS)]
    return aug(image)


def copy_images(image_paths, dest_dir):
    """
    Copy a list of images to *dest_dir*, preserving original filenames.

    Args:
        image_paths (list[str]): absolute paths to source images.
        dest_dir (str): absolute path to destination directory.
    """
    os.makedirs(dest_dir, exist_ok=True)
    for src in image_paths:
        shutil.copy2(src, dest_dir)


def augment_to_target(
    source_images,
    dest_dir,
    target_count,
    current_count,
):
    """
    Generate augmented images into *dest_dir* until *target_count*
    images exist there.

    Args:
        source_images (list[str]): originals to augment from.
        dest_dir (str): destination folder (already contains originals).
        target_count (int): desired total number of images.
        current_count (int): images already present in *dest_dir*.

    Returns:
        int: number of augmented images that were saved.
    """
    needed = target_count - current_count
    aug_saved = 0
    aug_index = 0

    while aug_saved < needed:
        src_path = source_images[aug_saved % len(source_images)]

        try:
            img = Image.open(src_path).convert('RGB')
            augmented = apply_augmentation(img, aug_index)

            name, ext = os.path.splitext(os.path.basename(src_path))
            filename = f"{name}_aug{aug_saved:04d}{ext}"
            augmented.save(os.path.join(dest_dir, filename))
            aug_saved += 1

        except Exception as exc:
            print(f"  Warning: skipped augmentation for "
                  f"{os.path.basename(src_path)}: {exc}")

        aug_index += 1

    return aug_saved


# ── main pipeline ─────────────────────────────────────────────────────────────


def process_dataset(input_dir):
    """
    Run the full balancing pipeline.

    - Discovers all leaf image categories under *input_dir*.
    - Splits each category into a test set (TEST_PER_CLASS images) and
      a training set.
    - Copies test images to   <parent>/test_images/<rel>/
    - Copies training images to <parent>/images_modified/<rel>/
    - Augments under-represented categories up to MAX_IMAGES.

    Args:
        input_dir (str): path to the root image directory.
    """
    input_dir = os.path.abspath(input_dir)
    parent_dir = os.path.dirname(input_dir)

    modified_dir = os.path.join(parent_dir, 'images_modified')
    test_dir = os.path.join(parent_dir, 'test_images')

    print(f"Input    : {input_dir}")
    print(f"Modified : {modified_dir}")
    print(f"Test     : {test_dir}")
    print(f"Target   : {MAX_IMAGES} images per category "
          f"(test: {TEST_PER_CLASS} each)\n")

    categories = collect_categories(input_dir)

    if not categories:
        print("No image categories found.")
        sys.exit(1)

    for rel_path, images in sorted(categories.items()):
        n_total = len(images)
        print(f"[{rel_path}]  original: {n_total}")

        random.shuffle(images)

        test_images = images[:TEST_PER_CLASS]
        train_images = images[TEST_PER_CLASS:]

        test_cat_dir = os.path.join(test_dir, rel_path)
        modified_cat_dir = os.path.join(modified_dir, rel_path)

        # ── test set ──────────────────────────────────────────────────
        copy_images(test_images, test_cat_dir)
        print(f"  test images copied   : {len(test_images)}")

        # ── training originals ────────────────────────────────────────
        copy_images(train_images, modified_cat_dir)
        current = len(train_images)
        print(f"  training originals   : {current}")

        # ── augmentation ─────────────────────────────────────────────
        if current < MAX_IMAGES:
            aug_count = augment_to_target(
                source_images=train_images,
                dest_dir=modified_cat_dir,
                target_count=MAX_IMAGES,
                current_count=current,
            )
            print(f"  augmented generated  : {aug_count}")
            print(f"  total in modified    : {current + aug_count}")
        else:
            print(f"  augmentation skipped "
                  f"(already >= {MAX_IMAGES})")

    print("\nDone.")
    print(f"  images_modified -> {modified_dir}")
    print(f"  test_images     -> {test_dir}")


# ── entry point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python balance_dataset.py <input_images_dir>")
        sys.exit(1)

    path = sys.argv[1]

    if not os.path.isdir(path):
        print(f"Error: '{path}' is not a valid directory.")
        sys.exit(1)

    process_dataset(path)
