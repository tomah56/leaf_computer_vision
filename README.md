# leaf_computer_vision

A machine learning-based image classification system for training and evaluating leaf-affliction (disease/pest) classifiers using PyTorch.

**Objectives:**
- Provide a simple, reproducible training pipeline for classifying leaf afflictions.
- Offer dataset utilities, augmentation and transform helpers to prepare image data.
- Include evaluation and visualization utilities to inspect model predictions and accuracy.

**Key achievements:**
- Modular core code in `leaffliction/core/` implementing dataset, model loading, metrics, and plotting.
- Training and inference scripts: `leaffliction/train.py` and `leaffliction/predict.py`.
- A pretrained model checkpoint is included at `model.pth` for quick inference and demos.
- Unit tests in `tests/` that exercise augmentation, dataset handling, and training code.

**Quickstart**
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare images: place your dataset under `leaffliction/images/` in class subfolders (one folder per class). Do NOT commit image data to the repository.

3. Train a model (example):

```bash
python3.10 leaffliction/train.py leaffliction/images/
```

This saves a checkpoint named `model_<foldername>.pth` after training.

4. Run prediction on a single image or using a JSON config:

```bash
python3.10 leaffliction/predict.py path/to/image.jpg
# or with a JSON config that lists model, folder and images
python3.10 leaffliction/predict.py config.json
```

**Project layout (high level):**
- `leaffliction/` — training, prediction, and dataset utilities.
- `leaffliction/core/` — dataset, model IO, transforms, metrics, plotting helpers.
- `tests/` — automated tests for core components.
- `model.pth` — example/trained model checkpoint (included for convenience).

If you want, I can also add a short example config file, improve the Quickstart with more flags, or run the test suite. Which would you like next?
