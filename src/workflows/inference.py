"""Basic infrastructure code to use the trained model 
to take in corrupted data and generate an uncorrupted sample."""
import sys
import os
sys.path.insert(1, os.path.join(os.getcwd(), 'src'))

import json
import yaml
import torch
import numpy as np
from core.models import ConvVAE
from core.dataset import setup, set_data_params
from tasks.plot import plot_input_output
import tasks.preprocessing as pre
from torch.utils.data import DataLoader
from dask.cache import Cache
from orchestration.constants import image_spec
import flytekit as fl
from flytekit.types.file import FlyteFile
from flytekit.extras.pytorch import PyTorchCheckpoint

torch.serialization.add_safe_globals([ConvVAE, torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear, torch.nn.modules.conv.ConvTranspose2d])
# comment these the next two lines out to disable Dask's cache
cache = Cache(1e10)  # 10gb cache
cache.register()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@fl.task(container_image=image_spec)
def generate_sample(model: ConvVAE, corrupted_sample: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        recon, _, _ = model(corrupted_sample)
    
    return recon

@fl.task(container_image=image_spec)
def plot_task(sample: torch.Tensor, corrupted_sample: torch.Tensor, recon: torch.Tensor, lons: np.ndarray, lats: np.ndarray) -> None:
    plot_input_output(sample.cpu(), corrupted_sample.cpu(), recon.cpu(), lons, lats)

@fl.task(container_image=image_spec, limits=fl.Resources(gpu="1"))
def load_model_checkpoint(checkpoint: FlyteFile) -> PyTorchCheckpoint:
    # Download file locally from FlyteBlob/S3/etc.
    local_path = checkpoint.download()
    print(local_path)
    # Load the model checkpoint
    model = torch.load(local_path, map_location=device, weights_only=False)
    print(type(model))

    # If needed: reinstantiate model class and load state_dict
    my_model = ConvVAE().to(device)
    my_model.load_state_dict(model)

    my_model.eval()
    
    return PyTorchCheckpoint(module=my_model)

@fl.task(container_image=image_spec, limits=fl.Resources(gpu="1"))
def load_model(checkpoint: PyTorchCheckpoint) -> ConvVAE:
    new_model = ConvVAE()
    new_model.load_state_dict(checkpoint["module_state_dict"])

    return new_model

@fl.task(container_image=image_spec, limits=fl.Resources(gpu="1"))
def check_for_gpu() -> bool:
    print(torch.cuda.is_available())
    return torch.cuda.is_available()

@fl.workflow
def infer() -> None:
    check_for_gpu()
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
    
    #diffusion_config = config['diffusion_params']
    #model_config = config['model_params']
    #train_config = config['train_params']
    #autoencoder_config = config['autoencoder_params']
    
    if os.path.exists(os.path.join("config", "min_max_dict.json")):
        with open(os.path.join("config", "min_max_dict.json"), "r") as f:
            min_max_dict = json.load(f)
        
        global_min = min_max_dict["global_min"]
        global_max = min_max_dict["global_max"]

    global_min = torch.tensor(min_max_dict["global_min"]).view(1, 1, 1, -1, 1, 1).to(device)
    global_max = torch.tensor(min_max_dict["global_max"]).view(1, 1, 1, -1, 1, 1).to(device)
    
    checkpoint = load_model_checkpoint(os.path.join("config", "cvae.pth"))

    model = load_model(checkpoint)

    sample = pre.sample_dataloader(training_generator)
    print(sample.shape)

    #sample = sample.float().to(device)
    sample_c = pre.corrupt_data(sample, 0.3)

    sample_norm = pre.scale_data(sample, global_min, global_max)
    sample_c_norm = pre.scale_data(sample_c, global_min, global_max, corrupted=True)

    sample_reshape = pre.reshape_batch(sample_norm)
    sample_c_reshape = pre.reshape_batch(sample_c_norm)

    recon = generate_sample(model, sample_c_reshape)

    print(recon.shape)

    plot_task(sample_reshape, sample_c_reshape, recon, dataset.lons, dataset.lats)


if __name__ == "__main__":
    infer()
