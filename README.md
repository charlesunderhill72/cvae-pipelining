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

```
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
```


## 🚀 Usage

### Setup

1. Clone this repo
2. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
3. Place your input .zarr or .nc data where the xbatcher.BatchGenerator can access it (configured in default.yaml).

## Training

Run the training pipeline:
```
python -m src.workflows.train_vae --num-epochs 20
```
This will:

* Load the dataset

* Generate global min/max stats (if not already saved)

* Train the CVAE for the specified number of epochs

* Save a model checkpoint to config/cvae.pth

