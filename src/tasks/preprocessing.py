"""Corrupt data and pad if necessary?"""

import os
import json
import yaml
import dask
import torch
import numpy as np
import xarray as xr
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.core.dataset import setup, set_data_params
from src.tasks.plot import plot_tensor_batch

def compute_global_min_max(dataloader: DataLoader, num_channels=2) -> dict:
    r"""
    Iterates over image attributes from Dataset class and returns global min and max.
    """
    global_min = [float('inf'), float('inf')]
    global_max = [float('-inf'), float('-inf')]

    for i, sample in tqdm(enumerate(dataloader), desc="Computing global min and max"):
       for j in range(num_channels):
          current_min = sample[:, :, :, j, :, :].min()
          current_max = sample[:, :, :, j, :, :].max()
          global_max[j] = max(global_max[j], current_max)
          global_min[j] = min(global_min[j], current_min)
    
    global_min_max = {
       "global_min": [float(value) for value in global_min],
       "global_max": [float(value) for value in global_max]
    }

    return global_min_max


def corrupt_data(tensor, k):
    tensor_c = tensor.detach().clone()
    m, n = int(tensor_c.shape[-2]), int(tensor_c.shape[-1])
    B, T, P, C = tensor_c.shape[:4]

    total_points = m * n
    num_corrupt = int(k * total_points)
    
    # Choose flat indices to corrupt
    indices = np.random.choice(total_points, size=num_corrupt, replace=False)
    lon_indices, lat_indices = np.unravel_index(indices, (m, n))

    # Create a corruption mask
    mask = torch.ones_like(tensor_c)
    mask[..., lon_indices, lat_indices] = 0

    tensor_c *= mask  # Apply mask

    return tensor_c


def scale_data(sample, global_min, global_max, device, corrupted=False):
    # Convert min/max lists to tensors and reshape for broadcasting
    #global_min = torch.tensor(min_max_dict["global_min"]).view(1, 1, 1, -1, 1, 1).to(device)
    #global_max = torch.tensor(min_max_dict["global_max"]).view(1, 1, 1, -1, 1, 1).to(device)
    # If sample is corrupted global_min == 0
    if corrupted:
        sample_norm = (sample) / (global_max)
    else:
        sample_norm = (sample - global_min) / (global_max - global_min)

    return sample_norm

def reshape_batch(sample):
    # Original shape: [B, T, P, C, H, W]
    B, T, P, C, H, W = sample.shape

    # Merge batch, time, and patch into a single batch dimension
    sample_vae_input = sample.view(B * T * P, C, H, W)

    return sample_vae_input

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Change to compute global min and max simultaneously, save computed values, and set to read in saved values if a save file exists
def main():
    # Read the config file #
    with open("config/default.yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    data_params = set_data_params(config)

    dataset = setup()
    training_generator = DataLoader(dataset, **data_params)

    if os.path.exists(os.path.join("config", "min_max_dict.json")):
       with open(os.path.join("config", "min_max_dict.json"), "r") as f:
            min_max_dict = json.load(f)
       
    else:
        min_max_dict = compute_global_min_max(training_generator)
        with open(os.path.join("config", "min_max_dict.json"), "w") as f:
            json.dump(min_max_dict, f)

    print("Global_min:", min_max_dict["global_min"])
    print("Global_max:", min_max_dict["global_max"])

    sample = next(iter(training_generator)).to(device)
    sample_c = corrupt_data(sample, 0.3).to(device)

    print(sample_c.shape)

    lons = dataset.lons
    lats = dataset.lats

    #plot_tensor_batch(sample_c, lons, lats, prefix="corrupted_data")

    # Min-max scaling
    sample_norm = scale_data(sample, min_max_dict, device)
    sample_c_norm = scale_data(sample_c, min_max_dict, device, corrupted=True)
    #plot_tensor_batch(sample_c_norm, lons, lats, prefix="scaled_c_data")

    sample_reshape = reshape_batch(sample_norm)

    #plot_tensor_batch(sample_reshape, lons, lats, prefix="batch")



if __name__ == '__main__':
   main()

