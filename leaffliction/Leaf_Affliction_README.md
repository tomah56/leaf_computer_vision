# 🍃 Leaf Affliction Classification Project

## 📌 Project Overview

This project implements an end-to-end computer vision pipeline for
classifying plant leaf diseases using deep learning.

The dataset consists of images organized in class-based folders:

    images/
     ├── Apple_Black_rot
     ├── Apple_healthy
     ├── Apple_rust
     ├── Apple_scab
     ├── Grape_Black_rot
     ├── Grape_Esca
     ├── Grape_healthy
     └── Grape_spot

Each folder name represents the ground-truth label for the images it
contains.

The final goal is to train a deep learning classifier achieving ≥90%
validation accuracy.

------------------------------------------------------------------------

## 🧠 Project Structure & Learning Objectives

The project is divided into multiple conceptual steps that follow a
traditional machine learning workflow:

### 1️⃣ Data Loading

-   Images are loaded using a PyTorch `Dataset` and `DataLoader`.
-   Folder names are automatically mapped to numerical class labels.
-   Images are resized and converted into tensors before training.

Example batch output:

    Images batch shape: torch.Size([4, 3, 224, 224])
    Labels batch: tensor([3, 0, 4, 1])

This confirms: - Batch size = 4 - 3 color channels (RGB) - Resolution =
224×224 - Labels correspond to folder classes

------------------------------------------------------------------------

### 2️⃣ Data Augmentation

Data augmentation techniques include:

-   Random rotations
-   Flips
-   Cropping
-   Color jitter

Purpose: - Improve model generalization - Reduce overfitting - Simulate
real-world variation - Address potential class imbalance

Note: In modern deep learning, especially when using transfer learning,
augmentation may not be strictly required to reach high accuracy on
clean datasets. However, it improves robustness and follows best
practices.

------------------------------------------------------------------------

### 3️⃣ Image Transformations & Analysis

This phase explores classical computer vision techniques such as:

-   Blurring
-   Histogram analysis
-   Masking
-   ROI extraction

Purpose: - Understand the dataset better - Explore visual
characteristics of the leaves - Develop intuition about image features

We can constrain or emphasize certain features by modifying input data, which influences what patterns the model learns.

for example:
-   Background causes bias → use ROI/mask.
-   Lighting varies in real life → use ColorJitter.
-   Leaves rotate naturally → use small rotation.

------------------------------------------------------------------------

### 4️⃣ Model Training (Deep Learning)

The final stage trains a convolutional neural network using PyTorch.

Typical setup: - Pretrained backbone (transfer learning) - Cross-entropy
loss - Adam optimizer - Train/validation split

Modern CNN architectures can often achieve high accuracy even without
heavy feature engineering due to automatic feature extraction.

------------------------------------------------------------------------

## 🔍 Why Some Steps May Feel "Optional"

With modern deep learning techniques:

    Data → Pretrained CNN → High Accuracy

This reduces the need for:

-   Manual feature engineering
-   Classical image processing pipelines
-   Extensive handcrafted transformations

However, the project intentionally follows a complete ML workflow to
ensure:

-   Understanding of data preprocessing
-   Awareness of overfitting
-   Knowledge of augmentation strategies
-   Ability to analyze dataset quality

------------------------------------------------------------------------

## 🎯 Key Learning Outcomes

By completing this project, we demonstrate:

-   PyTorch dataset and dataloader implementation
-   Image preprocessing and augmentation
-   Understanding of class-label mapping
-   Transfer learning with CNNs
-   Model evaluation and validation
-   Awareness of dataset robustness and bias

------------------------------------------------------------------------

## 🧩 Final Reflection

Although modern technology makes image classification significantly
easier than in the past, this project emphasizes understanding the full
pipeline rather than only achieving accuracy.

High accuracy alone is not sufficient --- understanding *why* the model
performs well is equally important.
