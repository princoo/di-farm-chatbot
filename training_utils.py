import torch
from torch import nn

def loss_fn() -> nn.Module:
    return torch.nn.CrossEntropyLoss()

def optimizer(model:nn.Module, learning_rate=0.001) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=learning_rate)
