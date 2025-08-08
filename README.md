# Geospatial CVAE Training Pipeline

This project implements a training pipeline for a **Convolutional Variational Autoencoder (CVAE)** on gridded geospatial data. The data is ingested using [xbatcher](https://xbatcher.readthedocs.io/) and loaded efficiently into PyTorch via a custom `Dataset` class with support for multi-process loading and lazy evaluation.

## 📌 Features

- Efficient batched data loading from large xarray datasets using `xbatcher`
- Dynamic preprocessing pipeline with:
  - Data corruption for denoising autoencoder training
  - Min-max normalization using global statistics
  - Reshaping of high-dimensional tensors for convolutional processing
- Modular training script with:
  - Multi-epoch training support
  - Automatic GPU utilization (if available)
  - Checkpointing to disk
- Optional multi-worker `DataLoader` for speedup
- Debugging and profiling hooks for memory usage and batch timing

## 📁 Project Structure

project/
│
├── config/
│ ├── default.yaml # Training and data loading config
│ ├── min_max_dict.json # Min/max stats for normalization
│ └── cvae.pth # Model checkpoint
│
├── src/
│ ├── workflows/
│ │ └── train_vae.py # Main training script (entry point)
│ ├── model/
│ │ └── vae.py # CVAE model definition
│ ├── data/
│ │ └── dataset.py # XBatcher-compatible PyTorch dataset
│ └── utils/
│ ├── preprocess.py # Preprocessing functions
│ └── helpers.py # Utility methods
│
└── README.md


