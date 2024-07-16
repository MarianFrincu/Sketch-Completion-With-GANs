import torch
import torch.nn as nn


class CustomTanh(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return (torch.tanh(x) + 1) / 2
