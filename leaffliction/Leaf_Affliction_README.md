# 🍃 Leaf Affliction Classification

## Project overview

This module provides a complete leaf disease classification workflow using
PyTorch and transfer learning.

Main capabilities:

- image dataset loading from folder structure,
- augmentation and preprocessing,
- model training with class balancing,
- evaluation and visualization,
- single-image and batch prediction.

## Expected data layout

The training and evaluation scripts expect class folders inside a root
data directory (for example [images](images)):

        images/
            apple/
                Apple_Black_rot/
                Apple_healthy/
                Apple_rust/
                Apple_scab/
            grape/
                Grape_Black_rot/
                Grape_Esca/
                Grape_healthy/
                Grape_spot/

Each leaf class folder is treated as one label.

## Quick start

From the [leaffliction](.) directory:

1. Install dependencies from [requirements.txt](requirements.txt).
2. Put training data under [images](images).
3. Train a model with [train.py](train.py).
4. Run predictions with [predict.py](predict.py), either:
     - with a JSON file (example: [config_example.json](config_example.json)), or
     - with a single JPG/PNG image path.

## Key files

- [train.py](train.py): model training loop and checkpoint saving.
- [predict.py](predict.py): inference entry point.
- [core/dataset.py](core/dataset.py): dataset discovery and dataset class.
- [core/model.py](core/model.py): model loading and prediction helpers.
- [core/transforms_utils.py](core/transforms_utils.py): train/eval transforms.
- [distribution.py](distribution.py): dataset distribution analysis utilities.
- [augmentation.py](augmentation.py): standalone augmentation script.
- [transformation.py](transformation.py): classical image processing tools.

## Notes

- Saved model checkpoints are committed in this folder for convenience.
- Raw image datasets should stay local and should not be committed.
