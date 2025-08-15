import sys
import os
sys.path.insert(1, os.path.join(os.getcwd(), 'src'))

import yaml
import typer
from typing import Optional
from typing_extensions import Annotated
import numpy as np
import torch
import torch.nn.functional as F
import json
import argparse
import tasks.preprocessing as pre
from torch.utils.data import DataLoader
from core.dataset import setup, set_data_params
from torch.optim import Adam
from tqdm import tqdm
from core.models import ConvVAE
from tasks.fit import final_loss
from dask.cache import Cache
from orchestration.constants import image_spec
import flytekit as fl

torch.serialization.add_safe_globals([ConvVAE, torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear, torch.nn.modules.conv.ConvTranspose2d, 
                                      torch.nn.modules.loss.BCELoss])

# comment these the next two lines out to disable Dask's cache
cache = Cache(1e10)  # 10gb cache
cache.register()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@fl.task(container_image=image_spec)
def configure_optimizer(model: ConvVAE, learning_rate: float) -> torch.optim.Adam:
    optimizer = Adam(model.parameters(), lr=learning_rate)

    return optimizer

@fl.task(container_image=image_spec)
def compute_bce(criterion: torch.nn.BCELoss, recon: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    return criterion(recon, sample)

@fl.task(container_image=image_spec)
def reconstruct_sample(model: ConvVAE, sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return model(sample)

@fl.task(container_image=image_spec)
def append_losses(losses: list[float], loss: torch.Tensor) -> None:
    losses.append(loss.item())
    
@fl.task(container_image=image_spec)
def print_loss(losses: list[float], loss: torch.Tensor, epoch_idx: int) -> None:
    losses.append(loss.detach().cpu().numpy())
    print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses),
        ))
    
@fl.task(container_image=image_spec)
def save_state(model: ConvVAE) -> None:
    torch.save(model.state_dict(), os.path.join("config",
                                                "cvae.pth"))

@fl.task(container_image=image_spec)
def optimize(loss: torch.Tensor, optimizer: torch.optim.Adam) -> None:
    loss.backward()
    optimizer.step()

@fl.task(container_image=image_spec)
def training_loop(model: ConvVAE, optimizer: torch.optim.Adam, criterion: torch.nn.BCELoss, num_epochs: int, training_generator: DataLoader, 
                  percent_corrupt: float, global_min: torch.Tensor, global_max: torch.Tensor) -> None:
    for epoch_idx in range(num_epochs):
        losses = []
        for i, sample in tqdm(enumerate(training_generator), total=len(training_generator)):
            #print(sample.shape)
            optimizer.zero_grad()
            sample = sample.float().to(device)
            sample_c = pre.corrupt_data(sample, percent_corrupt)

            sample_norm = pre.scale_data(sample, global_min, global_max)
            sample_c_norm = pre.scale_data(sample_c, global_min, global_max, corrupted=True)

            sample_reshape = pre.reshape_batch(sample_norm)
            sample_c_reshape = pre.reshape_batch(sample_c_norm)

            recon, mu, log_var = reconstruct_sample(model, sample_c_reshape)
            bce_loss = compute_bce(criterion, recon, sample_reshape)
            loss = final_loss(bce_loss, mu, log_var)
            append_losses(losses, loss)
            #losses.append(loss.item())
            optimize(loss, optimizer)
            #loss.backward()
            #optimizer.step()

            del sample, sample_c, sample_norm, sample_c_norm, sample_reshape, sample_c_reshape, recon, mu, log_var
            #torch.cuda.empty_cache()

        # This needs to be wrapped in a task
        #losses.append(loss.detach().cpu().numpy())
        #print('Finished epoch:{} | Loss : {:.4f}'.format(
        #        epoch_idx + 1,
        #        np.mean(losses),
        #    ))
        print_loss(losses, loss, epoch_idx)

        #torch.save(model.state_dict(), os.path.join("config",
        #                                        "cvae.pth"))
        save_state(model)


@fl.workflow
def train_autoencoder(num_epochs: Annotated[int, typer.Option(min=0, max=1000)] = 1,
                      percent_corrupt: Annotated[float, typer.Option(min=0, max=0.9)] = 0.3,
                      learning_rate: Annotated[float, typer.Option(min=10e-6, max=10e-2)] = 0.001) -> None:
    _locals = {k: v for k, v in locals().items() if not k.startswith("_")}
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
        min_max_dict = pre.compute_global_min_max(training_generator)
        with open(os.path.join("config", "min_max_dict.json"), "w") as f:
            json.dump(min_max_dict, f)
    
    global_min = torch.tensor(min_max_dict["global_min"]).view(1, 1, 1, -1, 1, 1).to(device)
    global_max = torch.tensor(min_max_dict["global_max"]).view(1, 1, 1, -1, 1, 1).to(device)

    # Instantiate the model
    model = ConvVAE().to(device)
    model.train()
    
    # Create output directories
    if not os.path.exists("config"):
        os.mkdir("config")
    
    # Load checkpoint if found
    if os.path.exists(os.path.join("config", "cvae.pth")):
        print('Loading checkpoint as found one')
        model.load_state_dict(torch.load(os.path.join("config",
                                                      "cvae.pth"), map_location=device))
    # Specify training parameters
    optimizer = configure_optimizer(model, learning_rate)
    criterion = torch.nn.BCELoss(reduction='sum')

    training_loop(model, optimizer, criterion, num_epochs, training_generator, percent_corrupt, global_min, global_max)


if __name__ == '__main__':
    typer.run(train_autoencoder)
