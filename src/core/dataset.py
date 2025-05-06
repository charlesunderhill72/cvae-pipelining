import json
import time
from typing import Optional

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

        # Use to_stacked_array to stack without broadcasting,
        stacked = batch.to_stacked_array(
            new_dim="batch", sample_dims=("time", "longitude", "latitude")
        ).transpose("time", "batch", ...)
        x = torch.tensor(stacked.data)
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
    
