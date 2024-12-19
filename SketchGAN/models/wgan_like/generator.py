import torch
import torch.nn as nn
from models.gan.modules.generator_module import GeneratorModule
from models.gan.modules.preprocessing_network import PreprocessingNetwork


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.preprocess_network_stage1 = PreprocessingNetwork(in_channels=in_channels)
        self.preprocess_network_stage2 = PreprocessingNetwork(in_channels=2 * in_channels)
        self.preprocess_network_stage3 = PreprocessingNetwork(in_channels=3 * in_channels)

        self.stage1 = GeneratorModule(in_channels, out_channels)
        self.stage2 = GeneratorModule(2 * in_channels, out_channels)
        self.stage3 = GeneratorModule(3 * in_channels, out_channels)

    def forward(self, x):
        stage1_input = self.preprocess_network_stage1(x)
        stage1_output = self.stage1(stage1_input)

        stage2_input = torch.cat([x, stage1_output], dim=1)
        stage2_input = self.preprocess_network_stage2(stage2_input)
        stage2_output = self.stage2(stage2_input)

        stage3_input = torch.cat([x, stage1_output, stage2_output], dim=1)
        stage3_input = self.preprocess_network_stage3(stage3_input)
        stage3_output = self.stage3(stage3_input)

        return stage3_output
