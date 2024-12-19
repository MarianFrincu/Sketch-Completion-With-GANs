import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, input_shape=None):
        super().__init__()

        if input_shape is None:
            input_shape = [1, 256, 256]  # [channels, height, width]

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.InstanceNorm2d(num_features=64, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.InstanceNorm2d(num_features=128, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.InstanceNorm2d(num_features=256, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            nn.InstanceNorm2d(num_features=512, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            nn.InstanceNorm2d(num_features=512, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            nn.InstanceNorm2d(num_features=512, affine=True),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            # the width and height is divided by 2**6 because before are applied 6 convolutions with stride 2
            nn.Linear(512 * (input_shape[1] // (2 ** 6)) * (input_shape[2] // (2 ** 6)), out_features=1024),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=1024, out_features=1)
        )

    def forward(self, x):
        return self.model(x)
