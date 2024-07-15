import torch
import torch.nn as nn
from models.modules.generator_module import GeneratorModule


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()

        self.preprocess_conv_stage1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                                kernel_size=1)
        self.preprocess_conv_stage2 = nn.Conv2d(in_channels=2 * in_channels, out_channels=2 * in_channels,
                                                kernel_size=1)
        self.preprocess_conv_stage3 = nn.Conv2d(in_channels=3 * in_channels, out_channels=3 * in_channels,
                                                kernel_size=1)

        self.stage1 = GeneratorModule(in_channels, out_channels, kernel_size, stride, padding)
        self.stage2 = GeneratorModule(2 * in_channels, out_channels, kernel_size, stride, padding)
        self.stage3 = GeneratorModule(3 * in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.preprocess_conv_stage1(x)
        x = self.preprocess_conv_stage1(x)
        y = self.stage1(x)

        x = torch.cat([x, y], dim=1)
        x = self.preprocess_conv_stage2(x)
        x = self.preprocess_conv_stage2(x)
        y = self.stage2(x)

        x = torch.cat([x, y], dim=1)
        x = self.preprocess_conv_stage3(x)
        x = self.preprocess_conv_stage3(x)
        y = self.stage3(x)

        return y
