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
├── config
│   ├── cvae.pth
│   ├── default.yaml
│   └── min_max_dict.json
├── docs
│   └── README.md
├── images
├── LICENSE
├── Notes.txt
├── preprocessing_test.txt
├── __pycache__
│   └── main.cpython-312.pyc
├── pyproject.toml
├── README.md
├── requirements.txt
├── src
│   ├── core
│   │   ├── dataset.py
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── __pycache__
│   │       ├── dataset.cpython-312.pyc
│   │       ├── __init__.cpython-312.pyc
│   │       └── models.cpython-312.pyc
│   ├── orchestration
│   │   ├── constants.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       ├── constants.cpython-312.pyc
│   │       ├── __init__.cpython-312.pyc
│   │       └── utils.cpython-312.pyc
│   ├── __pycache__
│   │   └── __init__.cpython-312.pyc
│   ├── tasks
│   │   ├── fit.py
│   │   ├── __init__.py
│   │   ├── plot.py
│   │   ├── preprocessing.py
│   │   ├── __pycache__
│   │   │   ├── fit.cpython-312.pyc
│   │   │   ├── __init__.cpython-312.pyc
│   │   │   ├── plot.cpython-312.pyc
│   │   │   └── preprocessing.cpython-312.pyc
│   │   └── test.py
│   └── workflows
│       ├── inference.py
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── inference.cpython-312.pyc
│       │   ├── __init__.cpython-312.pyc
│       │   └── train_vae.cpython-312.pyc
│       └── train_vae.py
├── test.txt
├── train_test.txt
└── uv.lock

```


