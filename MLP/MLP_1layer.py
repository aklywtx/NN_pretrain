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

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Create checkpoint directory
ckpt_dir = Path("checkpoints")
ckpt_dir.mkdir(exist_ok=True)

def save_checkpoint(model, epoch, loss):
    ckpt_path = ckpt_dir / f"model_epoch_{epoch}.pt"
    
    # Save the new checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, ckpt_path)
    
    # Keep only the latest checkpoint and the best checkpoint
    checkpoints = sorted(ckpt_dir.glob("model_epoch_*.pt"))
    if len(checkpoints) > 2:
        # Remove the oldest checkpoint
        checkpoints[0].unlink()

# To load a checkpoint later:
def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

### Data Generation ###

class DotDataset(Dataset):
    def __init__(self, image_dir='./display_dpi32', transform=None):
        self.image_dir = image_dir
        self.image_size = image_size
        self.transform = transform
        self.data = []
        self.labels = []
        
        # Get list of image files from the directory
        image_files = [f for f in os.listdir(image_dir) if f.startswith('image_')]
        
        for img_file in image_files:
            # Parse filename to get dots information
            # image_numBlackDots_numTotalDots_index.png
            parts = img_file.split('_')
            num_black = int(parts[1])
            num_total = int(parts[2])
            
            # Calculate probability
            prob = num_black / num_total
            
            # Load and process image
            img_path = os.path.join(image_dir, img_file)
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            image = np.array(image) / 255.0  # Normalize to [0,1]
            image = image.astype(np.float32).flatten()
            
            self.data.append(image)
            self.labels.append(prob)
            
        self.num_samples = len(self.data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return torch.tensor(sample, device=device), torch.tensor([label], dtype=torch.float32, device=device)

### Model Definition ###

class OneLayerMLP(nn.Module):
    def __init__(self, input_size):
        super(OneLayerMLP, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        out = self.fc(x)
        return out

### Hyperparameters ###

input_size = 256 * 256  # Flattened image size (256 * 256 pixels)
num_epochs = 20
batch_size = 64
learning_rate = 0.001

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

### Dataset and DataLoader ###

dataset = DotDataset(image_dir='display', image_size=28)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

### Initialize Model, Loss Function, and Optimizer ###

model = OneLayerMLP(input_size).to(device)  # Move model to GPU
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

### Training Loop ###
best_loss = float('inf')

# Add progress bar for epochs
for epoch in tqdm(range(num_epochs), desc='Epochs'):
    epoch_loss = 0.0
    # Add progress bar for batches
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
        # Move tensors to GPU
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    
    # Log metrics to wandb
    wandb.log({
        "epoch": epoch + 1,
        "loss": avg_loss,
    })
    
    # Save checkpoint every 5 epochs and on the final epoch
    if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
        save_checkpoint(model, epoch + 1, avg_loss)

# Finish wandb run
wandb.finish()