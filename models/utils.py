import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import wandb
import pandas as pd
from PIL import Image
import os

class CheckpointManager:
    def __init__(self, directory="./checkpoints"):
        self.ckpt_dir = Path(directory)
        self.ckpt_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(self, model, epoch, loss):
        ckpt_path = self.ckpt_dir / f"model_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss,
        }, ckpt_path)
        
        checkpoints = sorted(self.ckpt_dir.glob("model_epoch_*.pt"))
        if len(checkpoints) > 2:
            checkpoints[0].unlink()
    
    def load_checkpoint(self, model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']

# class DotDataset(Dataset):
#     def __init__(self, image_dir, for_vit=True, device='cpu'):
#         self.image_dir = image_dir
#         self.for_vit = for_vit
#         self.device = device
#         self.data = []
#         self.labels = []
        
#         image_files = [f for f in os.listdir(image_dir) if f.startswith('image_')]
        
    #     for img_file in image_files:
    #         parts = img_file.split('_')
    #         num_black = int(parts[1])
    #         num_total = int(parts[2])
    #         prob = num_black / num_total
            
    #         img_path = os.path.join(image_dir, img_file)
    #         image = Image.open(img_path).convert('RGB')  # Remove .convert('L')
    #         image = np.array(image) / 255.0  # Normalize to [0,1]
            
    #         if self.for_vit:
    #             image = image.transpose(2, 0, 1)  # Convert from [H, W, C] to [C, H, W] format for ViT
    #         else:
    #             image = image.flatten()  # Flatten for MLP
            
    #         self.data.append(image)
    #         self.labels.append(prob)
            
    #     self.data = np.array(self.data, dtype=np.float32)
    #     self.labels = np.array(self.labels, dtype=np.float32)

    # def __len__(self):
    #     return len(self.data)

    # def __getitem__(self, idx):
    #     return (torch.tensor(self.data[idx], device=self.device), 
    #             torch.tensor([self.labels[idx]], dtype=torch.float32, device=self.device))

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, device, model_type="vit"):
        self.image_paths = image_paths
        self.device = device
        self.model_type = model_type
        # self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = np.array(image) / 255.0
        if self.model_type == "vit":
            image = image.transpose(2, 0, 1)
        elif self.model_type == "mlp":
            image = image.flatten()
        # if self.transform:
        #     image = self.transform(image)
        image_name = self.image_paths[idx].split("/")[-1]
        num_black = int(image_name.split("_")[1])
        num_total = int(image_name.split("_")[2])
        label = num_black / num_total
        
        return (torch.tensor(image, dtype=torch.float32, device=self.device), torch.tensor(label, dtype=torch.float32, device=self.device))


class Trainer:
    def __init__(self, model, criterion, optimizer, checkpoint_manager, device, model_type="mlp", dropout=0, save_checkpoint=False, patience=10, scheduler=None, wandb=wandb):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_manager = checkpoint_manager
        self.device = device
        self.model_type = model_type
        assert model_type in ["mlp", "vit"]
        self.dropout = dropout
        self.save_checkpoint = save_checkpoint
        self.patience = patience
        self.scheduler = scheduler  # Add scheduler
    
    def train(self, train_loader, val_loader, num_epochs):
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in tqdm(range(num_epochs), desc='Epochs'):
            # Training phase
            epoch_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                # outputs = outputs.squeeze(1)
                loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                # print('Outputs shape:', outputs.shape)
                # print('Targets shape:', labels.shape)
            
            train_avg_loss = epoch_loss / len(train_loader)
            
            # Validation phase
            val_loss = self._validate(val_loader)
            
            # Step the scheduler after each epoch
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                wandb.log({"learning_rate": current_lr})
            
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_avg_loss,
                "val_loss": val_loss
            })
            
            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # if self.save_checkpoint:
                #     self.checkpoint_manager.save_checkpoint(self.model, epoch + 1, val_loss)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                    break
            
            # if self.save_checkpoint and ((epoch + 1) % 10 == 0 or epoch == num_epochs - 1):
            #     self.checkpoint_manager.save_checkpoint(self.model, epoch + 1, val_loss)
        

    
    def _validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def test(self, test_loader):
        self.model.eval()
        total_loss = 0
        predictions = []
        actual_values = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Testing'):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                actual_values.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
        self._analyze_results(avg_loss, predictions, actual_values)
    
    def _analyze_results(self, avg_loss, predictions, actual_values):
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        
        results_df = pd.DataFrame({
            'Actual': actual_values.flatten(),
            'Predicted': predictions.flatten()
        })
        results_df.to_csv(f'./results/test_results_{self.model_type}_dropout{self.dropout}.csv', index=False)
        
        mse = np.mean((predictions - actual_values) ** 2)
        mae = np.mean(np.abs(predictions - actual_values))
        
        print(f"\nTest Results:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        # Create scatter plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

        ax1.scatter(actual_values, predictions, color="blue", alpha=0.5)
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predictions')
        ax1.set_title('Predictions vs True Values')
        
        ax2.scatter(actual_values, predictions - actual_values, color="red", alpha=0.5)
        ax2.set_xlabel('True Values')
        ax2.set_ylabel('Predictions - True')
        ax2.set_title(f'Error vs True Values ({self.model_type}, dropout{self.dropout})')
        
        plt.tight_layout()
        plt.savefig(f'./plots/test_results_{self.model_type}_dropout{self.dropout}.png')
        plt.close()
