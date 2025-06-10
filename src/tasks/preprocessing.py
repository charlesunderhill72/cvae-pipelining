import os
import json
import yaml
import dask
import torch
import numpy as np
import xarray as xr
from torch.utils.data import DataLoader
from tqdm import tqdm
from core.dataset import setup, set_data_params
from tasks.plot import plot_tensor_batch
from orchestration.constants import image_spec
import flytekit as fl


@fl.task(container_image=image_spec)
def compute_global_min_max(dataloader: DataLoader, num_channels: int=2) -> dict:
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

@fl.task(container_image=image_spec)
def corrupt_data(tensor: torch.Tensor, k: float) -> torch.Tensor: 
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

    return tensor_c.float().to("cuda" if torch.cuda.is_available() else "cpu")

@fl.task(container_image=image_spec)
def scale_data(sample: torch.Tensor, global_min: torch.Tensor, global_max: torch.Tensor, corrupted: bool=False) -> torch.Tensor:
    # Convert min/max lists to tensors and reshape for broadcasting
    # If sample is corrupted global_min == 0
    if corrupted:
        sample_norm = (sample) / (global_max)
    else:
        sample_norm = (sample - global_min) / (global_max - global_min)

    return sample_norm

@fl.task(container_image=image_spec)
def reshape_batch(sample: torch.Tensor) -> torch.Tensor:
    # Original shape: [B, T, P, C, H, W]
    B, T, P, C, H, W = sample.shape

    # Merge batch, time, and patch into a single batch dimension
    sample_vae_input = sample.view(B * T * P, C, H, W)

    return sample_vae_input

@fl.task(requests=fl.Resources(mem="4Gi"), limits=fl.Resources(mem="8Gi"), container_image=image_spec)
def sample_dataloader(dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
    return next(iter(dataloader)).to(device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@fl.workflow
def main() -> None:
    # Read the config file #
    with open("config/default.yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    data_params = set_data_params(config)

    # Below line added to try and resolve sample generation using multiple workers with fl
    #os.environ["PYTHONPATH"] = os.path.abspath("src")  # or just "src" if you're sure of working dir

    dataset = setup()
    training_generator = DataLoader(dataset, **data_params)

    if os.path.exists(os.path.join("config", "min_max_dict.json")):
       with open(os.path.join("config", "min_max_dict.json"), "r") as f:
            min_max_dict = json.load(f)
       
    else:
        min_max_dict = compute_global_min_max(training_generator)
        with open(os.path.join("config", "min_max_dict.json"), "w") as f:
            json.dump(min_max_dict, f)

    global_min = torch.tensor(min_max_dict["global_min"]).view(1, 1, 1, -1, 1, 1).to(device)
    global_max = torch.tensor(min_max_dict["global_max"]).view(1, 1, 1, -1, 1, 1).to(device)

    print("Global_min:", global_min)
    print("Global_max:", global_max)

    sample = sample_dataloader(training_generator)
    sample_c = corrupt_data(sample, 0.3)

    #print(sample_c.shape)
    print(os.getcwd())

    lons = dataset.lons
    lats = dataset.lats

    plot_tensor_batch(sample_c, lons, lats, save_dir="images_test", prefix="corrupted_data")

    # Min-max scaling
    sample_norm = scale_data(sample, global_min, global_max)
    sample_c_norm = scale_data(sample_c, global_min, global_max, corrupted=True)
    #plot_tensor_batch(sample_c_norm, lons, lats, prefix="scaled_c_data")

    sample_reshape = reshape_batch(sample_norm)

    #plot_tensor_batch(sample_reshape, lons, lats, prefix="batch")



if __name__ == '__main__':
   main()

