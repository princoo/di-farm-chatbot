import torch
from torch import nn

class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_units,output_size):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_size,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=output_size),
            # nn.ReLU(),
        )

    def forward(self,X):
       return self.layer_stack(X)
    

