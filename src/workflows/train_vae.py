
"""import os
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import json
import argparse
from torch.utils.data import DataLoader
from dataset.mnist_dataset import MnistDataset
from torch.optim import Adam
from tqdm import tqdm
from models.autoencoder_base import ConvAutoencoder, final_loss
from utils.external_utilities import compute_global_min_max, next_power_of_2, pad_to_power_of_2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_autoencoder(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    autoencoder_config = config['autoencoder_params']
    dataset_config = config['dataset_params']
    train_config = config['train_params']
    
    if os.path.exists(os.path.join(train_config['task_name'], train_config['min_max_name'])):
        with open(os.path.join(train_config['task_name'], train_config['min_max_name']), "r") as f:
            min_max_dict = json.load(f)
        
        global_min = min_max_dict["global_min"]
        global_max = min_max_dict["global_max"]
    
    else:
        # Load image paths using temp Dataset instance
        temp_dataset = MnistDataset(split="train", im_path=dataset_config['im_path'], global_min=0, global_max=1)
        image_paths = temp_dataset.images
        
        # Compute global min and max
        global_min, global_max = compute_global_min_max(image_paths)
        print(f"Global min: {global_min}, max: {global_max}")
        
        min_max_dict = {
            "global_min": float(global_min),
            "global_max": float(global_max)
        }
        
        with open(os.path.join(train_config['task_name'], train_config['min_max_name']), "w") as f:
            json.dump(min_max_dict, f)
    
    # Create the dataset
    mnist = MnistDataset('train', im_path=dataset_config['im_path'], global_min=global_min, global_max=global_max)
    mnist_loader = DataLoader(mnist, batch_size=autoencoder_config['batch_size'], shuffle=True, num_workers=4)
    
    # Instantiate the model
    model = ConvAutoencoder().to(device)
    model.train()
    
    # Create output directories
    if not os.path.exists(autoencoder_config['task_name']):
        os.mkdir(autoencoder_config['task_name'])
    
    # Load checkpoint if found
    if os.path.exists(os.path.join(autoencoder_config['task_name'],autoencoder_config['ckpt_name'])):
        print('Loading checkpoint as found one')
        model.load_state_dict(torch.load(os.path.join(autoencoder_config['task_name'],
                                                      autoencoder_config['ckpt_name']), map_location=device))
    # Specify training parameters
    num_epochs = autoencoder_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=autoencoder_config['lr'])
    criterion = torch.nn.MSELoss()
    
    model.to(device)
    model.train()

    for epoch_idx in range(num_epochs):
        losses = []
        for im in tqdm(mnist_loader):
            optimizer.zero_grad()
            im = im.float().to(device)
            im = pad_to_power_of_2(im)

            recon, mu, log_var = model(im)
            mse_loss = criterion(recon, im)
            loss = final_loss(mse_loss, mu, log_var)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        print('Finished epoch:{} | Loss : {:.4f}'.format(
                epoch_idx + 1,
                np.mean(losses),
            ))

        torch.save(model.state_dict(), os.path.join(autoencoder_config['task_name'],
                                                autoencoder_config['ckpt_name']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for autoencoder training')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    train_autoencoder(args)"""



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
    training_generator = DataLoader(dataset, **data_params)
    _ = next(iter(training_generator))  # wait until dataloader is ready
    t1 = time.time()
    print_json({"event": "setup end", "time": t1, "duration": t1 - t0})

    for epoch in range(num_epochs):
        e0 = time.time()
        print_json({"event": "epoch start", "epoch": epoch, "time": e0})

        for i, sample in enumerate(training_generator):
            tt0 = time.time()
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

