"""The goal for this module is a flexible plotting tool."""

import io
import os
import dask
import yaml
import xbatcher
import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from PIL import Image
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from torch.utils.data import DataLoader
from core.dataset import setup, set_data_params
from orchestration.constants import image_spec
from flytekit.types.file import PNGImageFile
from flytekitplugins.deck.renderer import ImageRenderer
import flytekit as fl


# Add a function to make plots from batch generator
@fl.task(container_image=image_spec)
def plot_batch_from_bgen(batch: xr.Dataset, save_path: str="./images/patches.png") -> None:
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

@fl.task(enable_deck=True, container_image=image_spec)
def plot_input_output(tensor_in: torch.Tensor, tensor_c: torch.Tensor, tensor_out: torch.Tensor, lons: np.ndarray, 
                      lats: np.ndarray, save_dir: str="./images", prefix: str="input_output") -> None:
    if (tensor_in.shape != tensor_out.shape):
        raise ValueError("Input and Output image must have same shape.")

    batch_size, num_levels, _, _ = tensor_in.shape

    for b in range(batch_size):
        fig, axes = plt.subplots(num_levels, 3, figsize=(8*num_levels, 20),
                                subplot_kw={'projection': ccrs.PlateCarree()})
        for l in range(num_levels):
            ax_or = axes[l, 0]
            ax_or.set_title(f"Original Batch {b}, Level {l}")
            ax_or.coastlines()
            cf = ax_or.contourf(lons, lats, tensor_in[b, l].numpy().T, 20, cmap="viridis")
            plt.colorbar(cf, ax=ax_or)

            ax_in = axes[l, 1]
            ax_in.set_title(f"Corrupted Batch {b}, Level {l}")
            ax_in.coastlines()
            cf = ax_in.contourf(lons, lats, tensor_c[b, l].numpy().T, 20, cmap="viridis")
            plt.colorbar(cf, ax=ax_in)

            ax_out = axes[l, 2]
            ax_out.set_title(f"Reconstructed Batch {b}, Level {l}")
            ax_out.coastlines()
            cf = ax_out.contourf(lons, lats, tensor_out[b, l].numpy().T, 20, cmap="viridis")
            plt.colorbar(cf, ax=ax_out)

        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{prefix}_b{b}.png")
        # Save figure to an in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        # Convert buffer to PIL Image
        image = Image.open(buf)
            
        # Use ImageRenderer to generate HTML and attach to deck
        html += ImageRenderer().to_html(image_src=image)
        plt.savefig(path)
        plt.close(fig)
            
    fl.Deck("Matplotlib Plot", html)


@fl.task(enable_deck=True, container_image=image_spec)
def plot_tensor_batch(tensor: torch.Tensor, lons: np.ndarray, lats: np.ndarray, save_dir: str="images", prefix: str="batch_timestep") -> None:
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
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"{prefix}_b{b}.png")
            # Save figure to an in-memory buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)

            # Convert buffer to PIL Image
            image = Image.open(buf)

            # Use ImageRenderer to generate HTML and attach to deck
            html += ImageRenderer().to_html(image_src=image)
            plt.savefig(path)
            plt.close(fig)

        fl.Deck("Matplotlib Plot", html)

    else:
        batch_size, time_steps, patches, num_levels, _, _ = tensor.shape

        html = ""
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
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, f"{prefix}_b{b}_t{t}.png")
                # Save figure to an in-memory buffer
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)

                # Convert buffer to PIL Image
                image = Image.open(buf)

                # Use ImageRenderer to generate HTML and attach to deck
                html += ImageRenderer().to_html(image_src=image)
                plt.savefig(path)
                plt.close(fig)

        fl.Deck("Matplotlib Plot", html)

@fl.task(container_image=image_spec)
def create_bgen(input_steps: int=3) -> xbatcher.BatchGenerator:
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

    return bgen

@fl.task(container_image=image_spec)
def sample_bgen(bgen: xbatcher.BatchGenerator) -> xr.Dataset:
    return next(iter(bgen))

@fl.task(requests=fl.Resources(mem="4Gi"), limits=fl.Resources(mem="8Gi"), container_image=image_spec)
def sample_dataloader(dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
    return next(iter(dataloader)).to(device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@fl.workflow
def main(input_steps: int=3) -> None:
    bgen = create_bgen(input_steps)

    sample_batch = sample_bgen(bgen)
    print(sample_batch)
    print(type(sample_batch))
    #plot_batch_from_bgen(sample_batch)

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
    
    sample = sample_dataloader(training_generator)
    lons = dataset.lons
    lats = dataset.lats
    print(type(lats))

    plot_tensor_batch(sample, lons, lats)






if __name__ == '__main__':
    main()

