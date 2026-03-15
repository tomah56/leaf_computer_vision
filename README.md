# Leaf Computer Vision

Leaf disease classification project built with PyTorch.

## Overview

This repository contains an end-to-end pipeline for:

- loading plant leaf image datasets from class folders,
- training a CNN classifier (ResNet-18 transfer learning),
- evaluating model accuracy,
- running predictions on single images or batches from a JSON config.

The core implementation lives in [leaffliction](leaffliction).

## Quick start

1. Create and activate a Python environment.
2. Install dependencies from [leaffliction/requirements.txt](leaffliction/requirements.txt).
3. Place dataset folders under [leaffliction/images](leaffliction/images).
4. Train a model from the [leaffliction](leaffliction) directory.
5. Run predictions with [leaffliction/predict.py](leaffliction/predict.py).

## Dataset location

Expected dataset root:

- [leaffliction/images](leaffliction/images)

Do not commit raw image datasets to git.

## Documentation

Detailed project documentation is available in
[leaffliction/Leaf_Affliction_README.md](leaffliction/Leaf_Affliction_README.md).
