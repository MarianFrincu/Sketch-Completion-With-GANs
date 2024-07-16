import torch
import torch.nn as nn
from util.custom_tanh import CustomTanh


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=4, stride=2, padding=1, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size, stride, padding, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=4, stride=2, padding=1, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, kernel_size, stride, padding, normalize=False)
        self.down2 = UNetDown(64, 128, kernel_size, stride, padding)
        self.down3 = UNetDown(128, 256, kernel_size, stride, padding)
        self.down4 = UNetDown(256, 512, kernel_size, stride, padding, dropout=0.5)
        self.down5 = UNetDown(512, 512, kernel_size, stride, padding, dropout=0.5)
        self.down6 = UNetDown(512, 512, kernel_size, stride, padding, dropout=0.5)
        self.down7 = UNetDown(512, 512, kernel_size, stride, padding, dropout=0.5)
        self.down8 = UNetDown(512, 512, kernel_size, stride, padding, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, kernel_size, stride, padding, dropout=0.5)
        self.up2 = UNetUp(1024, 512, kernel_size, stride, padding, dropout=0.5)
        self.up3 = UNetUp(1024, 512, kernel_size, stride, padding, dropout=0.5)
        self.up4 = UNetUp(1024, 512, kernel_size, stride, padding, dropout=0.5)
        self.up5 = UNetUp(1024, 256, kernel_size, stride, padding)
        self.up6 = UNetUp(512, 128, kernel_size, stride, padding)
        self.up7 = UNetUp(256, 64, kernel_size, stride, padding)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, kernel_size, padding=1),
            CustomTanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)
