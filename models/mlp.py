import torch.nn as nn
import torch

# class OneLayerMLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, dropout=0):
#         super(OneLayerMLP, self).__init__()
#         self.hidden = nn.Linear(input_size, hidden_size)
#         self.output = nn.Linear(hidden_size, output_size)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=dropout)

#         # self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.relu(self.hidden(x))
#         x = self.dropout(x)
#         x = self.output(x)
#         # out = self.sigmoid(out)
#         return x
   
class OneLinear(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0):
        super(OneLinear, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        return x
     
class TwoLinear(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0):
        super(TwoLinear, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
    
class TwoLinear(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0):
        super(TwoLinear, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
    
class ReLUMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0):
        super(ReLUMLP, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        return x

class ReLUMLP_withGaussianNoise(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0, noise_std=0.1):
        super(ReLUMLP_withGaussianNoise, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)
        self.noise_std = noise_std  # Standard deviation of Gaussian noise

    def forward(self, x):
        # Add Gaussian noise to the input
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        x = self.linear(x)
        x = self.relu(x)
        
        # Add Gaussian noise to the hidden representation if needed
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        x = self.dropout(x)
        x = self.linear2(x)
        
        return x