import torch.nn as nn

class OneLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0):
        super(OneLayerMLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.dropout(x)
        x = self.output(x)
        # out = self.sigmoid(out)
        return x