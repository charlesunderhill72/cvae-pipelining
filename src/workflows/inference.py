"""Basic infrastructure code to use the trained model 
to take in corrupted data and generate an uncorrupted sample."""
import os
import json
import yaml
import torch
from src.core.models import ConvVAE
from src.core.dataset import setup, set_data_params
from src.tasks.plot import plot_input_output
import src.tasks.preprocessing as pre
from torch.utils.data import DataLoader
from dask.cache import Cache

# comment these the next two lines out to disable Dask's cache
cache = Cache(1e10)  # 10gb cache
cache.register()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def infer():
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
    
    # Load autoencoder with checkpoint
    model = ConvVAE().to(device)
    model.load_state_dict(torch.load(os.path.join("config",
                                                  "cvae.pth"), map_location=device))
    model.eval()

    sample = next(iter(training_generator))
    print(sample.shape)

    sample = sample.float().to(device)
    sample_c = pre.corrupt_data(sample, 0.3).float().to(device)

    sample_norm = pre.scale_data(sample, global_min, global_max)
    sample_c_norm = pre.scale_data(sample_c, global_min, global_max, corrupted=True)

    sample_reshape = pre.reshape_batch(sample_norm)
    sample_c_reshape = pre.reshape_batch(sample_c_norm)

    with torch.no_grad():
        recon, _, _ = model(sample_c_reshape)

    print(recon.shape)

    plot_input_output(sample_reshape.cpu(), sample_c_reshape.cpu(), recon.cpu(), dataset.lons, dataset.lats)


if __name__ == "__main__":
    infer()
