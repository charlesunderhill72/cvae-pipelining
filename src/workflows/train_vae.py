
import os
import yaml
import typer
from typing import Optional
from typing_extensions import Annotated
import numpy as np
import torch
import torch.nn.functional as F
import json
import argparse
import src.tasks.preprocessing as pre
from torch.utils.data import DataLoader
from src.core.dataset import setup
from torch.optim import Adam
from tqdm import tqdm
from src.core.models import ConvVAE
from src.tasks.fit import final_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_autoencoder(num_epochs: Annotated[int, typer.Option(min=0, max=1000)] = 2):
    _locals = {k: v for k, v in locals().items() if not k.startswith("_")}
    # Read the config file #
    with open("config/default.yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    data_params = pre.set_data_params(config)

    dataset = setup()
    training_generator = DataLoader(dataset, **data_params)
    #print(len(training_generator))
    
    #autoencoder_config = config['autoencoder_params']
    #dataset_config = config['dataset_params']
    #train_config = config['train_params']
    
    if os.path.exists(os.path.join("config", "min_max_dict.json")):
       with open(os.path.join("config", "min_max_dict.json"), "r") as f:
            min_max_dict = json.load(f)
       
    else:
        min_max_dict = pre.compute_global_min_max(training_generator)
        with open(os.path.join("config", "min_max_dict.json"), "w") as f:
            json.dump(min_max_dict, f)
    
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
    num_epochs = num_epochs
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss(reduction='sum')
    losses = []

    for epoch_idx in range(num_epochs):
        for i, sample in tqdm(enumerate(training_generator), total=len(training_generator)):
            print(sample.shape)
            optimizer.zero_grad()
            sample = sample.float().to(device)
            sample_c = pre.corrupt_data(sample, 0.3).float().to(device)

            sample_norm = pre.scale_data(sample, min_max_dict, device)
            sample_c_norm = pre.scale_data(sample_c, min_max_dict, device, corrupted=True)

            sample_reshape = pre.reshape_batch(sample_norm)
            sample_c_reshape = pre.reshape_batch(sample_c_norm)

            recon, mu, log_var = model(sample_c_reshape)
            bce_loss = criterion(recon, sample_reshape)
            loss = final_loss(bce_loss, mu, log_var)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        losses.append(loss.detach().cpu().numpy())
        print('Finished epoch:{} | Loss : {:.4f}'.format(
                epoch_idx + 1,
                loss.item(),
            ))

        torch.save(model.state_dict(), os.path.join("config",
                                                "cvae.pth"))


if __name__ == '__main__':
    typer.run(train_autoencoder)
