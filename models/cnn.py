import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.optim import Adam

# Define a ResNet-based regression model
class ResNetRegression(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetRegression, self).__init__()
        # Load pre-trained ResNet model
        self.resnet = resnet18(pretrained=pretrained)
        
        # Modify the first layer for grayscale images (optional)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the fully connected layer with a regression layer
        num_ftrs = self.resnet.fc.in_features  # Get the number of input features to the last layer
        self.resnet.fc = nn.Linear(num_ftrs, 1)  # Output 1 value for regression

    def forward(self, x):
        return self.resnet(x)

# Instantiate the model
model = ResNetRegression(pretrained=True)