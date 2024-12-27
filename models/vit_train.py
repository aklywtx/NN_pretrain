from utils import CheckpointManager, Trainer, CustomImageDataset
from vit import ViT
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

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")


def main():
    image_size = 256
    num_epochs = 50
    batch_size = 64
    learning_rate = 0.00001
    image_size = 256
    patch_size = 16
    dim = 64
    depth = 1
    heads = 1
    mlp_dim = 256
    model_type = "vit"
    run_id = 1
    dropout = 0
    emb_dropout = 0

    # Initialize wandb
    wandb.init(
        project=f"{model_type}-dots",
        name=f"noise-{run_id}",
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "image_size": image_size,
            "dim": dim,
            "dropout": dropout,
        },
        dir="./wandb_results",
    )
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=1,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        pool="mean",
        channels=1,
        dim_head=dim,
        dropout=dropout,
        emb_dropout=emb_dropout,
    ).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # # Add Cosine Annealing scheduler
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=num_epochs,  # Total number of epochs
    #     eta_min=1e-6      # Minimum learning rate
    # )

    checkpoint_manager = CheckpointManager()

    # Create dataset and dataloader
    train_dir = "./training_data/displays_dpi32_only100dots"
    val_dir = "./val_data/displays_dpi32_MLPval_only100dots"
    test_dir = "./test_data/displays_dpi32_MLPtest_only100dots"
    train_paths = glob.glob(os.path.join(train_dir, "*.png"))
    val_paths = glob.glob(os.path.join(val_dir, "*.png"))
    test_paths = glob.glob(os.path.join(test_dir, "*.png"))
    train_dataset = CustomImageDataset(
        image_paths=train_paths, device=device, model_type=model_type
    )
    val_dataset = CustomImageDataset(
        image_paths=val_paths, device=device, model_type=model_type
    )
    test_dataset = CustomImageDataset(
        image_paths=test_paths, device=device, model_type=model_type
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # # Use glob to find all image files matching the pattern
    # train_dir = '/Users/aklywtx/Desktop/VLM_Bias_research/NN_pretrain/training_data/displays_dpi32_ViT_only100dots/'
    # val_dir = '/Users/aklywtx/Desktop/VLM_Bias_research/NN_pretrain/val_data/displays_dpi32_ViTval_only100dots'
    # test_dir = '/Users/aklywtx/Desktop/VLM_Bias_research/NN_pretrain/test_data/displays_dpi32_ViTtest_only100dots'
    # train_paths = glob.glob(os.path.join(train_dir, '*.png'))
    # val_paths = glob.glob(os.path.join(val_dir, '*.png'))
    # test_paths = glob.glob(os.path.join(test_dir, '*.png'))

    # # train_dir = './training_data/displays_dpi32_only100dots'
    # # val_dir = './val_data/displays_dpi32_MLPval_only100dots'
    # # test_dir = './test_data/displays_dpi32_MLPtest_only100dots'
    # train_dataset = CustomImageDataset(train_paths, device=device)
    # val_dataset = CustomImageDataset(val_paths, device=device)
    # test_dataset = CustomImageDataset(test_paths, device=device)
    # # train_dataset = DotDataset(image_dir=train_dir, for_vit=True, device=device)
    # # val_dataset = DotDataset(image_dir=val_dir, for_vit=True, device=device)
    # # test_dataset = DotDataset(image_dir=test_dir, for_vit=True, device=device)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # for (i, j) in test_loader:
    #     print(i.size(), j.size())
    #     break

    # Create trainer and run training
    trainer = Trainer(
        model,
        criterion,
        optimizer,
        checkpoint_manager,
        device,
        model_type=model_type,
        dropout=dropout,
        patience=10,
        run_id=run_id,
    )
    trainer.train(train_loader, val_loader, num_epochs)
    trainer.test(test_loader)

    wandb.finish()


if __name__ == "__main__":
    main()