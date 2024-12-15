from utils import CheckpointManager, CustomImageDataset, Trainer
from mlp import OneLayerMLP
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb
import random
import glob
import os

file_name = __file__.split("/")[-1]

torch.manual_seed(42)
np.random.seed(42)
file_id = random.randint(0, 10000)

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

def main():  
    input_size = 256 * 256 * 3
    num_epochs = 100
    batch_size = 64
    learning_rate = 1e-4
    hidden_size = 256
    output_size = 1
    dropout = 0
    model_type="mlp"
    
    # Initialize wandb
    wandb.init(
        project=f"{model_type}-dots",
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "input_size": input_size,
            "dropout": dropout,
        },
        dir="./wandb_results"
    )
    
    # Initialize components
    checkpoint_manager = CheckpointManager()
    model = OneLayerMLP(input_size, hidden_size, output_size).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create dataset and dataloader
    train_dir = './training_data/displays_dpi32_only100dots'
    val_dir = './val_data/displays_dpi32_MLPval_only100dots'
    test_dir = './test_data/displays_dpi32_MLPtest_only100dots'
    train_paths = glob.glob(os.path.join(train_dir, '*.png'))
    val_paths = glob.glob(os.path.join(val_dir, '*.png'))
    test_paths = glob.glob(os.path.join(test_dir, '*.png'))
    train_dataset = CustomImageDataset(image_paths=train_paths, device=device, model_type=model_type)
    val_dataset = CustomImageDataset(image_paths=val_paths, device=device, model_type=model_type)
    test_dataset = CustomImageDataset(image_paths=test_paths, device=device, model_type=model_type)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # Create trainer and run training
    trainer = Trainer(model, criterion, optimizer, checkpoint_manager, device, model_type=model_type, dropout=dropout, patience=10)
    trainer.train(train_loader, val_loader, num_epochs)
    trainer.test(test_loader)
    
    wandb.finish()
    
if __name__ == "__main__":
    main()