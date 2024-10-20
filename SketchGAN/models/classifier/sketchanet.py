import torch.nn as nn


class SketchANet(nn.Module):
    def __init__(self, in_channels=3, num_classes=125):
        super().__init__()

        self.model = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=15, stride=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # Layer 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # Layer 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Layer 4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Layer 5
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # Layer 6
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=7, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Layer 7
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Layer 8
            nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1, stride=1, padding=0),
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x)

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
