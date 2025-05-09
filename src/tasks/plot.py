"""The goal for this module is a flexible plotting tool."""

import os
import dask
import xbatcher
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from src.core.dataset import setup, XBatcherPyTorchDataset

def set_data_params(config):
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

    dask_threads = params.get("dask_threads")
    if dask_threads is None or dask_threads <= 1:
        dask.config.set(scheduler="single-threaded")
    else:
        dask.config.set(scheduler="threads", num_workers=dask_threads)

    return data_params

# Add a function to make plots from batch generator
def plot_batch_from_bgen(batch, save_path="patches.png"):
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


def plot_tensor_batch(tensor, lats, lons, save_dir=".", prefix="timestep"):
    time_steps, batch_size, num_levels, _, _ = tensor.shape

    for t in range(time_steps):
        fig, axes = plt.subplots(batch_size, num_levels, figsize=(6*num_levels, 4*batch_size),
                                 subplot_kw={'projection': ccrs.PlateCarree()})
        for b in range(batch_size):
            for l in range(num_levels):
                ax = axes[b, l] if batch_size > 1 else axes[l]
                ax.set_title(f"Time {t}, Batch {b}, Level {l}")
                ax.coastlines()
                cf = ax.contourf(lons, lats, tensor[t, b, l].numpy(), 20, cmap="viridis")
                plt.colorbar(cf, ax=ax)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_t{t}.png"))
        plt.close(fig)


def create_image(save_dir, fname, tensor, lons, lats):
    # Make num_rows = number of batches (tensor.shape[1]) and num_columns = number of levels (tensor.shape[2])
    f, ax = plt.subplots(2, 2, figsize=(18, 14), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set up the map
    ax[0, 0].set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())  # Global extent

    # Add natural features
    ax[0, 0].add_feature(cfeature.COASTLINE)
    ax[0, 0].add_feature(cfeature.BORDERS, linestyle=':')

    # Create contour plot
    contour50 = ax[0, 0].contourf(lons, lats, X_test_norm[0], 20, transform=ccrs.PlateCarree(), cmap='viridis')

    # Add a color bar
    cbar = plt.colorbar(contour50, ax=ax[0, 0], orientation='vertical', shrink=0.75)
    cbar.set_label('Geopotential (m^2 s^-2)')

    # Add title
    ax[0, 0].set_title('Original Geopotential Contour Map 50mb', fontsize=12)

    # Set up the map
    ax[0, 1].set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())  # Global extent

    # Add natural features
    ax[0, 1].add_feature(cfeature.COASTLINE)
    ax[0, 1].add_feature(cfeature.BORDERS, linestyle=':')

    # Create contour plot
    contour50_rec = ax[0, 1].contourf(lons, lats, test_rec_np2[0][0], 20, transform=ccrs.PlateCarree(), cmap='viridis')

    # Add a color bar
    cbar = plt.colorbar(contour50_rec, ax=ax[0, 1], orientation='vertical', shrink=0.75)
    cbar.set_label('Geopotential (m^2 s^-2)')

    # Add title
    ax[0, 1].set_title('Reconstructed Geopotential Contour Map 50mb', fontsize=12)

    # Set up the map
    ax[1, 0].set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())  # Global extent

    # Add natural features
    ax[1, 0].add_feature(cfeature.COASTLINE)
    ax[1, 0].add_feature(cfeature.BORDERS, linestyle=':')

    # Create contour plot
    contour50 = ax[1, 0].contourf(lons, lats, X_test_norm[1], 20, transform=ccrs.PlateCarree(), cmap='viridis')

    # Add a color bar
    cbar = plt.colorbar(contour50, ax=ax[1, 0], orientation='vertical', shrink=0.75)
    cbar.set_label('Geopotential (m^2 s^-2)')

    # Add title
    ax[1, 0].set_title('Original Geopotential Contour Map 500mb', fontsize=12)

    # Set up the map
    ax[1, 1].set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())  # Global extent

    # Add natural features
    ax[1, 1].add_feature(cfeature.COASTLINE)
    ax[1, 1].add_feature(cfeature.BORDERS, linestyle=':')

    # Create contour plot
    contour50_rec = ax[1, 1].contourf(lons, lats, test_rec_np2[0][1], 20, transform=ccrs.PlateCarree(), cmap='viridis')

    # Add a color bar
    cbar = plt.colorbar(contour50_rec, ax=ax[1, 1], orientation='vertical', shrink=0.75)
    cbar.set_label('Geopotential (m^2 s^-2)')

    # Add title
    ax[1, 1].set_title('Reconstructed Geopotential Contour Map 500mb', fontsize=12)

    f.savefig(os.path.join(save_dir, fname))


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




if __name__ == '__main__':
    main()

