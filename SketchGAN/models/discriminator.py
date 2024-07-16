import torch
import torch.nn as nn
from models.global_discriminator import GlobalDiscriminator
from models.local_discriminator import LocalDiscriminator
from util.custom_tanh import CustomTanh


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.global_discriminator = GlobalDiscriminator()
        self.local_discriminator = LocalDiscriminator()
        self.fully_connected = nn.Linear(2048, 1)
        self.activation = nn.ReLU()
        self.custom_tanh = CustomTanh()

    def forward(self, x, y):
        x = self.global_discriminator(x)
        y = self.local_discriminator(y)
        y_x_concat = torch.cat([y, x], dim=1)

        y = self.fully_connected(y_x_concat)
        y = self.activation(y)

        return self.custom_tanh(y)
