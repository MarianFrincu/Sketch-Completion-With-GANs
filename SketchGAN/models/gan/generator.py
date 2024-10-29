import torch
import torch.nn as nn
from models.gan.modules.generator_module import GeneratorModule
from models.gan.modules.preprocessing_network import PreprocessingNetwork


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.preprocess_network_stage1 = PreprocessingNetwork(in_channels=in_channels, out_channels=in_channels)
        self.preprocess_network_stage2 = PreprocessingNetwork(in_channels=2 * in_channels, out_channels=2 * in_channels)
        self.preprocess_network_stage3 = PreprocessingNetwork(in_channels=3 * in_channels, out_channels=3 * in_channels)

        self.stage1 = GeneratorModule(in_channels, out_channels)
        self.stage2 = GeneratorModule(2 * in_channels, out_channels)
        self.stage3 = GeneratorModule(3 * in_channels, out_channels)

    def forward(self, x):
        x = self.preprocess_network_stage1(x)
        y = self.stage1(x)

        x = torch.cat([x, y], dim=1)
        x = self.preprocess_network_stage2(x)
        y = self.stage2(x)

        x = torch.cat([x, y], dim=1)
        x = self.preprocess_network_stage3(x)
        y = self.stage3(x)

        return y
