import numpy as np
import pandas as pd
import torch
from torch import nn
import lightning as L
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from lightning.pytorch.loggers import WandbLogger
from introduction_code import GaussianFit, MSELoss_batch
from typing import Union

import sys
sys.path.append('./Convolutional-KANs/kan_convolutional')
from KANLinear import *

# Setup Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Setup Randomness
L.seed_everything(628, workers=True)

# CUDA Efficiency
torch.set_float32_matmul_precision('high')

# Logging
import wandb
wandb.login(key='2a6309b336d51d2918fc4fb2d51ffef9505c370a')

########
# DATA #
########
class LCDataset(Dataset):
    def __init__(self, dataset, num_classes, limit=-1):
        self.limit = limit
        self.num_classes = num_classes
        if self.limit != -1:
            sub = list(np.random.permutation(np.arange(len(dataset)))[0:self.limit])
            self.dataset = Subset(dataset, sub)
        else:
            self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        y_one_hot = torch.zeros(self.num_classes)
        y_one_hot[y] = 1
        return x, y_one_hot

# Data transforms for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 datasets
train_dataset = CIFAR10("./temp/", train=True, download=True, transform=transform)
test_dataset = CIFAR10("./temp/", train=False, download=True, transform=transform)

# Split train dataset into train and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Dataloader function
num_workers = 10
def get_dataloader(split: Union["train", "val", "test", "ntk"]="train", batch_size=64, limit=-1):
    data_loader = None
    if split == "ntk":
        data_loader = DataLoader(
            LCDataset(train_dataset, num_classes=10, limit=500), 
            batch_size=batch_size, num_workers=num_workers
        )
    elif split == "train":
        data_loader = DataLoader(
            LCDataset(train_dataset, num_classes=10, limit=limit), 
            batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
    elif split == "val":
        data_loader = DataLoader(
            LCDataset(val_dataset, num_classes=10, limit=limit), 
            batch_size=batch_size, num_workers=num_workers
        )
    elif split == "test":
        data_loader = DataLoader(
            LCDataset(test_dataset, num_classes=10, limit=500), 
            batch_size=batch_size, num_workers=num_workers
        )
    return data_loader

#########
# Model #
#########
class ClassicKAN(L.LightningModule):
    def __init__(self, num_hidden_layers=1, hidden_dim=64, inp_size=32*32*3, out_size=10, grid_size=2, spline_order=2):
        super().__init__()
        self.inp_size = inp_size  # Changed to match CIFAR-10 dimensions (32x32x3)
        layers_hidden = [inp_size] + [hidden_dim] * num_hidden_layers + [out_size]
        self.net = nn.Sequential(
            KAN(layers_hidden=layers_hidden, grid_size=grid_size, spline_order=spline_order)
        )
    
    def forward(self, x):
        x = x.view(-1, self.inp_size)  # Flatten the 32x32x3 input
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        v1 = torch.argmax(y_pred, dim=1)
        v2 = torch.argmax(y, dim=1)
        accuracy = torch.sum(torch.eq(v1, v2)) / len(y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        v1 = torch.argmax(y_pred, dim=1)
        v2 = torch.argmax(y, dim=1)
        accuracy = torch.sum(torch.eq(v1, v2)) / len(y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        v1 = torch.argmax(y_pred, dim=1)
        v2 = torch.argmax(y, dim=1)
        accuracy = torch.sum(torch.eq(v1, v2)) / len(y)
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

def check_ntk_acc(model, dataloader):
    res = 0.0
    sumlength = 0
    model.eval()
    model.to(device)
    for it in iter(dataloader):
        x, y = it
        x = x.to(device)
        y = y.to(device)
        sumlength += len(x)
        res += (torch.argmax(model.forward(x), dim=1) == torch.argmax(y, dim=1)).sum()
    model.train()
    return res / sumlength

#########
# Train #
#########
# Set sweep parameters
sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    "parameters": {
        "epochs": {
            "values": [8]
        },
        "learning_rate": {
            "values": [0.001, 0.01]
        },
        "batch_size": {
            "values": [64]
        },
        "num_hidden": {
            "values": [1, 2, 3]
        },
        "hidden_dim": {
            "values": [16, 64, 256]
        },
        "limit": {
            "values": [32, 256, 1024]
        },
        "grid_size": {
            "values": [2]
        },
        "spline_order": {
            "values": [2]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="KANTK_CIFAR10")  # Changed project name

def run_wandb(config=None):
    with wandb.init(config=config):
        config = wandb.config
        wandb.run.name = f"Run-{wandb.run.id}"
        
        # Load data
        train_loader = get_dataloader(split="train", batch_size=config.batch_size, limit=-1)
        val_loader = get_dataloader(split="val", batch_size=config.batch_size, limit=-1)
        test_loader = get_dataloader(split="test", batch_size=config.batch_size)
        ntk_loader = get_dataloader(split="ntk", batch_size=config.batch_size, limit=config.limit)
        
        # Initialize model
        model = ClassicKAN(
            num_hidden_layers=config.num_hidden,
            hidden_dim=config.hidden_dim,
            inp_size=32*32*3,  # Changed to match CIFAR-10 dimensions
            grid_size=config.grid_size,
            spline_order=config.spline_order
        ).to(device)
        model.net.to(device)

        # Log model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.log({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        })

        # Train and test model
        logger = WandbLogger(project="KANTK_CIFAR10", log_model="all")  # Changed project name
        trainer = L.Trainer(
            max_epochs=config.epochs,
            deterministic=True,
            logger=logger, 
            val_check_interval=1
        )
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)
        
        # Compute NTK accuracy
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        ntk_model = GaussianFit(model=model, device=device, noise_var=0.0)
        ntk_model.fit(ntk_loader, optimizer, MSELoss_batch)
        ntk_acc = check_ntk_acc(model, test_loader)
        wandb.log({"ntk_accuracy": ntk_acc.item()})
        
        # Log model predictions
        example_batch = next(iter(test_loader))
        example_images, example_labels = example_batch
        predictions_KAN = model(example_images.to(device)).argmax(dim=1)
        predictions_NTKAN = ntk_model(example_images.to(device)).argmax(dim=1)
        
        # Log test examples
        wandb.log({
            "examples_KAN": [
                wandb.Image(example_images[i], caption=f"Pred: {predictions_KAN[i]}, Label: {example_labels[i]}")
                for i in range(len(example_images))
            ],
            "examples_NTKAN": [
                wandb.Image(example_images[i], caption=f"Pred: {predictions_NTKAN[i]}, Label: {example_labels[i]}")
                for i in range(len(example_images))
            ]
        })

if __name__ == "__main__":
    wandb.agent(sweep_id, run_wandb)