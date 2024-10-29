import torch
import torch.nn as nn

from models.gan.modules.global_discriminator import GlobalDiscriminator
from models.gan.modules.local_discriminator import LocalDiscriminator


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.global_discriminator = GlobalDiscriminator()
        self.local_discriminator = LocalDiscriminator()
        self.fully_connected = nn.Linear(2048, 1)
        self.activation = nn.ReLU()

    def forward(self, x, y):
        x = self.global_discriminator(x)
        y = self.local_discriminator(y)

        z = self.fully_connected(torch.cat([y, x], dim=1))
        z = self.activation(z)

        return (torch.tanh(z) + 1) / 2
