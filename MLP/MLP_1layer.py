import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import wandb
from pathlib import Path
from tqdm import tqdm
import pandas as pd

torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ### Checkpoint Management ###
# class CheckpointManager:
#     def __init__(self, directory="checkpoints"):
#         self.ckpt_dir = Path(directory)
#         self.ckpt_dir.mkdir(exist_ok=True)
    
#     def save_checkpoint(self, model, epoch, loss):
#         ckpt_path = self.ckpt_dir / f"model_epoch_{epoch}.pt"
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'loss': loss,
#         }, ckpt_path)
        
#         # Keep only the latest checkpoint and the best checkpoint
#         checkpoints = sorted(self.ckpt_dir.glob("model_epoch_*.pt"))
#         if len(checkpoints) > 2:
#             checkpoints[0].unlink()
    
#     def load_checkpoint(self, model, checkpoint_path):
#         checkpoint = torch.load(checkpoint_path)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         return checkpoint['epoch'], checkpoint['loss']



class OneLayerMLP(nn.Module):
    def __init__(self, input_size):
        super(OneLayerMLP, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        out = self.fc(x)
        return out

### Training and Testing Functions ###
class MLPTrainer:
    def __init__(self, model, criterion, optimizer, checkpoint_manager, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_manager = checkpoint_manager
        self.device = device
    
    def train(self, train_loader, num_epochs):
        self.model.train()
        best_loss = float('inf')
        
        for epoch in tqdm(range(num_epochs), desc='Epochs'):
            epoch_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            wandb.log({"epoch": epoch + 1, "loss": avg_loss})
            
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                self.checkpoint_manager.save_checkpoint(self.model, epoch + 1, avg_loss)
    
    def test(self, test_dir, batch_size):
        self.model.eval()
        test_dataset = DotDataset(image_dir=test_dir)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
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
        
        self._analyze_results(total_loss / len(test_loader), predictions, actual_values)
    
    def _analyze_results(self, avg_loss, predictions, actual_values):
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        
        # Save predictions and actual values to CSV
        results_df = pd.DataFrame({
            'Actual': actual_values.flatten(),
            'Predicted': predictions.flatten()
        })
        results_df.to_csv('test_results.csv', index=False)
        
        mse = np.mean((predictions - actual_values) ** 2)
        mae = np.mean(np.abs(predictions - actual_values))
        
        print(f"\nTest Results:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Results saved to test_results.csv")
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(actual_values, predictions - actual_values, alpha=0.5)
        # plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('true Values')
        plt.ylabel('Predictions - true')
        plt.title('Error vs true values')
        plt.savefig('test_results.png')
        plt.close()

### Main Execution ###
def main():
    # Hyperparameters
    input_size = 256 * 256
    num_epochs = 6
    batch_size = 64
    learning_rate = 0.0001
    
    # Initialize wandb
    wandb.init(
        project="mlp-dots",
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "input_size": input_size,
        }
    )
    
    # Initialize components
    checkpoint_manager = CheckpointManager()
    model = OneLayerMLP(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create dataset and dataloader
    dataset = DotDataset(image_dir='/local/xtong/NN_pretrain/training_data/displays_dpi32')
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create trainer and run training
    trainer = MLPTrainer(model, criterion, optimizer, checkpoint_manager, device)
    trainer.train(train_loader, num_epochs)
    
    # Test the model
    test_dir = '/local/xtong/NN_pretrain/test_data/displays_dpi32_MLPtest'
    trainer.test(test_dir, batch_size)
    
    wandb.finish()

if __name__ == "__main__":
    main()
