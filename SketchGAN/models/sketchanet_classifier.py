import torch
import torch.nn as nn


class SketchANet(torch.nn.Module):
    def __init__(self, in_channels=3, num_classes=125):
        super().__init__()

        self.model = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 64, (15, 15), stride=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=2, padding=0),
            # Layer 2
            nn.Conv2d(64, 128, (5, 5), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=2, padding=0),
            # Layer 3
            nn.Conv2d(128, 256, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            # Layer 4
            nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            # Layer 5
            nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=2, padding=0),
            # Layer 6
            nn.Conv2d(256, 512, (7, 7), stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Layer 7
            nn.Conv2d(512, 512, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Layer 8
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)
