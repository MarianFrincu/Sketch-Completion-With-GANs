import torch.nn as nn


class LocalDiscriminator(nn.Module):
    def __init__(self, input_shape=None, kernel_size=5, stride=2, padding=2):
        super().__init__()

        if input_shape is None:
            input_shape = [3, 128, 128]  # [channels, height, width]

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=64, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),
            # the width and height is divided by 2**5 because before are applied 5 convolutions with stride 2
            nn.Linear(512 * (input_shape[1] // (2 ** 5)) * (input_shape[2] // (2 ** 5)), 1024),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)
