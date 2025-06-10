import json
import time
from typing import Optional
import numpy as np

import dask
import torch
import typer
import xbatcher
import xarray as xr
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch import multiprocessing
from typing_extensions import Annotated

from dask.cache import Cache

# comment these the next two lines out to disable Dask's cache
cache = Cache(1e10)  # 10gb cache
cache.register()


def print_json(obj):
    print(json.dumps(obj))


class XBatcherPyTorchDataset(TorchDataset):
    def __init__(self, batch_generator: xbatcher.BatchGenerator):
        self.bgen = batch_generator
        example = self.bgen[0].load()
        self.lons = example.longitude.values.copy()
        self.lats = example.latitude.values.copy()

    def __len__(self):
        return len(self.bgen)

    def __getitem__(self, idx):
        t0 = time.time()
        print_json(
            {
                "event": "get-batch start",
                "time": t0,
                "idx": idx,
                "pid": multiprocessing.current_process().pid,
            }
        )
        # load before stacking
        batch = self.bgen[idx].load()
        #self.lons = batch.longitude.values
        #self.lats = batch.latitude.values

        # Print coordinate ranges
        #print("Latitudes:", batch.latitude.values)
        #print("Longitudes:", batch.longitude.values)
        #print("Time:", batch.time.values)

        # get min/max
        #print("Lat range:", batch.latitude.values.min(), "to", batch.latitude.values.max())
        #print("Lon range:", batch.longitude.values.min(), "to", batch.longitude.values.max())

        # Use to_stacked_array to stack without broadcasting,
        stacked = batch.to_stacked_array(
            new_dim="batch", sample_dims=("time", "level", "longitude", "latitude")
        ).transpose("time", "batch", ...)
        x = torch.tensor(stacked.data)
        #print(id(x))
        #print("test")
        #print(x.shape)
        t1 = time.time()
        print_json(
            {
                "event": "get-batch end",
                "time": t1,
                "idx": idx,
                "pid": multiprocessing.current_process().pid,
                "duration": t1 - t0,
            }
        )
        return x
    

def setup(source="gcs", split="1979", set="train",  patch_size_lon: int = 64, patch_size_lat: int = 32, input_steps: int = 1):
    if source == "gcs":
        ds = xr.open_dataset(
            "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr",
            engine="zarr",
            chunks={},
        )

    else:
        raise ValueError(f"Unknown source {source}")

    DEFAULT_VARS = [
        "geopotential"
    ]

    start_time = np.datetime64('1959-01-01T00:00:00')
    train_split_time = np.datetime64(f'{int(split)-1}-12-31T18:00:00')
    test_split_time = np.datetime64(f'{split}-01-01T00:00:00')
    end_time = np.datetime64('2023-01-10T18:00:00')

    if set == "train":
        ds = ds[DEFAULT_VARS].sel(time=slice(start_time, train_split_time), level=[50, 500])

    elif set == "test":
        ds = ds[DEFAULT_VARS].sel(time=slice(test_split_time, end_time), level=[50, 500])

    else:
        raise ValueError(f"set parameter must be either train or test, not {set}")
    
    patch = dict(
        level=2,
        longitude=patch_size_lon,
        latitude=patch_size_lat,
        time=input_steps,
    )
    overlap = dict(level=0, longitude=0, latitude=0, time=input_steps // 3 * 2) # Overlap in time dimension to capture temporal correlations

    bgen = xbatcher.BatchGenerator(
        ds,
        input_dims=patch,
        input_overlap=overlap,
        preload_batch=False,
    )
    #print(ds)

    dataset = XBatcherPyTorchDataset(bgen)

    return dataset

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
        data_params["multiprocessing_context"] = params["multiprocessing_context"]
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

def main(
    source: Annotated[str, typer.Option()] = "gcs",
    num_epochs: Annotated[int, typer.Option(min=0, max=1000)] = 2,
    num_batches: Annotated[int, typer.Option(min=0, max=1000)] = 3,
    batch_size: Annotated[int, typer.Option(min=0, max=1000)] = 16,
    shuffle: Annotated[Optional[bool], typer.Option()] = None,
    num_workers: Annotated[Optional[int], typer.Option(min=0, max=64)] = None,
    prefetch_factor: Annotated[Optional[int], typer.Option(min=0, max=64)] = None,
    persistent_workers: Annotated[Optional[bool], typer.Option()] = None,
    pin_memory: Annotated[Optional[bool], typer.Option()] = None,
    train_step_time: Annotated[Optional[float], typer.Option()] = 0.1,
    dask_threads: Annotated[Optional[int], typer.Option()] = None,
):
    _locals = {k: v for k, v in locals().items() if not k.startswith("_")}
    data_params = {
        "batch_size": batch_size,
    }
    if shuffle is not None:
        data_params["shuffle"] = shuffle
    if num_workers is not None:
        data_params["num_workers"] = num_workers
        data_params["multiprocessing_context"] = "forkserver"
    if prefetch_factor is not None:
        data_params["prefetch_factor"] = prefetch_factor
    if persistent_workers is not None:
        data_params["persistent_workers"] = persistent_workers
    if pin_memory is not None:
        data_params["pin_memory"] = pin_memory
    if dask_threads is None or dask_threads <= 1:
        dask.config.set(scheduler="single-threaded")
    else:
        dask.config.set(scheduler="threads", num_workers=dask_threads)

    run_start_time = time.time()
    print_json(
        {
            "event": "run start",
            "time": run_start_time,
            "data_params": str(data_params),
            "locals": _locals,
        }
    )

    t0 = time.time()
    print_json({"event": "setup start", "time": t0})
    dataset = setup(source=source)
    print("Dataset Length:", len(dataset))
    training_generator = DataLoader(dataset, **data_params)
    _ = next(iter(training_generator))  # wait until dataloader is ready
    t1 = time.time()
    print_json({"event": "setup end", "time": t1, "duration": t1 - t0})

    for epoch in range(num_epochs):
        e0 = time.time()
        print_json({"event": "epoch start", "epoch": epoch, "time": e0})

        for i, sample in enumerate(training_generator):
            tt0 = time.time()
            print(torch.mean(torch.mean(sample, dim=2)))
            print_json({"event": "training start", "batch": i, "time": tt0})
            time.sleep(train_step_time)  # simulate model training
            tt1 = time.time()
            print_json({"event": "training end", "batch": i, "time": tt1, "duration": tt1 - tt0})
            if i == num_batches - 1:
                break

        e1 = time.time()
        print_json({"event": "epoch end", "epoch": epoch, "time": e1, "duration": e1 - e0})

    run_finish_time = time.time()
    print_json(
        {"event": "run end", "time": run_finish_time, "duration": run_finish_time - run_start_time}
    )



if __name__ == '__main__':
    typer.run(main)