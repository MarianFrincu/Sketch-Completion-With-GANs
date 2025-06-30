import torch
import torch.nn as nn

from models.global_discriminator import GlobalDiscriminator
from models.local_discriminator import LocalDiscriminator


class Discriminator(nn.Module):
    def __init__(self, global_shape=None, local_shape=None):
        super().__init__()

        if global_shape is None:
            global_shape = [1, 256, 256]

        if local_shape is None:
            local_shape = [1, 128, 128]

        self.global_discriminator = GlobalDiscriminator(input_shape=global_shape)
        self.local_discriminator = LocalDiscriminator(input_shape=local_shape)
        self.fully_connected = nn.Linear(2048, 1)


    def forward(self, x, y):
        x = self.global_discriminator(x)
        y = self.local_discriminator(y)

        z = self.fully_connected(torch.cat([y, x], dim=1))

        return (torch.tanh(z) + 1) / 2
