"""Corrupt data and pad if necessary?"""

import os
import json
import yaml
import dask
import numpy as np
import xarray as xr
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.core.dataset import XBatcherPyTorchDataset, setup

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

def compute_global_max(dataloader: DataLoader, num_channels=2) -> list[float]:
    r"""
    Iterates over image attributes from Dataset class and returns global max.
    """
    global_max = [float('-inf'), float('-inf')]

    for i, sample in tqdm(enumerate(dataloader), desc="Computing global max"):
       for j in range(num_channels):
          current_max = sample[:, :, :, j, :, :].max()
          global_max[j] = max(global_max[j], current_max)
    
    return global_max


def corrupt_data(data, k):
  """Gives corrupted data for either one timestep or for a full year dataset
     depending on the input. k is the percent of corrupted data points.
     Expects a numpy array for the data input."""
  data_c = np.copy(data)
  m, n = 91, 180
  indices = np.random.choice(m*n, size=int(k*m*n), replace=False)

  # Convert the indices into row and column indices in the 2D meshgrid
  lat_indices, lon_indices = np.unravel_index(indices, (m, n))

  # Set the randomly selected points to 0
  if (np.size(data_c.shape) == 3):
    for i in range(2):
      data_c[i][lat_indices, lon_indices] = 0

  elif (np.size(data_c.shape) == 4):
    for i in range(data_c.shape[0]):
      for j in range(2):
        indices = np.random.choice(m*n, size=int(k*m*n), replace=False)
        # Convert the indices into row and column indices in the 2D meshgrid
        lat_indices, lon_indices = np.unravel_index(indices, (m, n))
        data_c[i][j][lat_indices, lon_indices] = 0

  return data_c


def set_data_params(config):
    """Read data params in from YAML config file."""
    params = config.get("data_params", {})
    data_params = {
        "batch_size": params.get("batch_size", 16),  # default fallback
    }

    if params.get("shuffle") is not None:
        data_params["shuffle"] = params["shuffle"]
    if params.get("num_workers") is not None:
        data_params["num_workers"] = params["num_workers"]
        data_params["multiprocessing_context"] = "forkserver"
    if params.get("prefetch_factor") is not None:
        data_params["prefetch_factor"] = params["prefetch_factor"]
    if params.get("persistent_workers") is not None:
        data_params["persistent_workers"] = params["persistent_workers"]
    if params.get("pin_memory") is not None:
        data_params["pin_memory"] = params["pin_memory"]

    dask_threads = params["dask_threads"]
    if dask_threads is None or dask_threads <= 1:
        dask.config.set(scheduler="single-threaded")
    else:
        dask.config.set(scheduler="threads", num_workers=dask_threads)

    return data_params

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



if __name__ == '__main__':
   main()

