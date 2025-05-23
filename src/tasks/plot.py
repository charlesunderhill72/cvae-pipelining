"""The goal for this module is a flexible plotting tool."""

import os
import dask
import yaml
import xbatcher
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from torch.utils.data import DataLoader
from src.core.dataset import setup, XBatcherPyTorchDataset, set_data_params


# Add a function to make plots from batch generator
def plot_batch_from_bgen(batch, save_path="./images/patches.png"):
    times = batch.time.values
    levels = batch.level.values
    num_levels = len(levels)
    num_patches = batch.sizes["time"]

    fig, axes = plt.subplots(num_patches, num_levels, figsize=(6*num_levels, 4*num_patches),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    for i, time in enumerate(times):
        for j, level in enumerate(levels):
            ax = axes[i, j] if num_patches > 1 else axes[j]

            patch_data = batch.sel(time=time, level=level)["geopotential"]
            lats = patch_data.latitude.values
            lons = patch_data.longitude.values
            z = patch_data.values.T                               # Need to look into why this has to be transposed

            ax.set_title(f"Patch {time}, Level {level}")
            ax.coastlines()
            cf = ax.contourf(lons, lats, z, 20, cmap="viridis")
            plt.colorbar(cf, ax=ax)

    plt.tight_layout()
    fig.savefig(save_path)


def plot_tensor_batch(tensor, lons, lats, save_dir="./images", prefix="batch_timestep"):
    if tensor.dim() == 4:
        batch_size, num_levels, _, _ = tensor.shape

        for b in range(batch_size):
            fig, axes = plt.subplots(num_levels, figsize=(6*num_levels, 18),
                                    subplot_kw={'projection': ccrs.PlateCarree()})
            for l in range(num_levels):
                ax = axes[l]
                ax.set_title(f"Batch {b}, Level {l}")
                ax.coastlines()
                cf = ax.contourf(lons, lats, tensor[b, l].numpy().T, 20, cmap="viridis")
                plt.colorbar(cf, ax=ax)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}_b{b}.png"))
            plt.close(fig)

    else:
        batch_size, time_steps, patches, num_levels, _, _ = tensor.shape

        for b in range(batch_size):
            for t in range(time_steps):
                fig, axes = plt.subplots(patches, num_levels, figsize=(6*num_levels, 4*patches),
                                        subplot_kw={'projection': ccrs.PlateCarree()})
                for p in range(patches):
                    for l in range(num_levels):
                        ax = axes[p, l] if patches > 1 else axes[l]
                        ax.set_title(f"Time {t}, Patch {p}, Level {l}")
                        ax.coastlines()
                        cf = ax.contourf(lons, lats, tensor[b, t, p, l].numpy().T, 20, cmap="viridis")
                        plt.colorbar(cf, ax=ax)

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{prefix}_b{b}_t{t}.png"))
                plt.close(fig)

def main(input_steps=3):
    ds = xr.open_dataset(
            "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr",
            engine="zarr",
            chunks={},
        )
    
    DEFAULT_VARS = [
        "geopotential"
    ]

    ds = ds[DEFAULT_VARS].sel(level=[50, 500])
    patch = dict(
        level=2,
        latitude=32,
        longitude=64,
        time=input_steps,
    )
    overlap = dict(level=0, latitude=0, longitude=0, time=input_steps // 3 * 2) # Overlap in time dimension to capture temporal correlations

    bgen = xbatcher.BatchGenerator(
        ds,
        input_dims=patch,
        input_overlap=overlap,
        preload_batch=False,
    )

    sample_batch = next(iter(bgen))
    print(sample_batch)
    plot_batch_from_bgen(sample_batch)

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
    
    sample = next(iter(training_generator))
    lons = dataset.lons
    lats = dataset.lats

    plot_tensor_batch(sample, lons, lats)






if __name__ == '__main__':
    main()

