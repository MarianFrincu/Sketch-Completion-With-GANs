import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, mode, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True,
                 activation='relu', dropout=0.0):
        super().__init__()
        layers = []

        if mode == 'encoder':
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
        elif mode == 'decoder':
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if 0.0 < dropout < 1.0:
            layers.append(nn.Dropout(dropout))

        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leaky':
            layers.append(nn.LeakyReLU(0.2))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class GeneratorModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down1 = Block(mode='encoder', in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1,
                           batch_norm=False, activation='leaky', dropout=0.0)
        self.down2 = Block(mode='encoder', in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1,
                           batch_norm=True, activation='leaky', dropout=0.0)
        self.down3 = Block(mode='encoder', in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1,
                           batch_norm=True, activation='leaky', dropout=0.0)
        self.down4 = Block(mode='encoder', in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1,
                           batch_norm=True, activation='leaky', dropout=0.0)
        self.down5 = Block(mode='encoder', in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1,
                           batch_norm=True, activation='leaky', dropout=0.0)
        self.down6 = Block(mode='encoder', in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1,
                           batch_norm=True, activation='leaky', dropout=0.0)
        self.down7 = Block(mode='encoder', in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1,
                           batch_norm=True, activation='leaky', dropout=0.0)
        self.down8 = Block(mode='encoder', in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1,
                           batch_norm=True, activation='leaky', dropout=0.0)

        self.up1 = Block(mode='decoder', in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1,
                         batch_norm=True, activation='relu', dropout=0.5)
        self.up2 = Block(mode='decoder', in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1,
                         batch_norm=True, activation='relu', dropout=0.5)
        self.up3 = Block(mode='decoder', in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1,
                         batch_norm=True, activation='relu', dropout=0.5)
        self.up4 = Block(mode='decoder', in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1,
                         batch_norm=True, activation='relu', dropout=0.0)
        self.up5 = Block(mode='decoder', in_channels=1024, out_channels=256, kernel_size=4, stride=2, padding=1,
                         batch_norm=True, activation='relu', dropout=0.0)
        self.up6 = Block(mode='decoder', in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1,
                         batch_norm=True, activation='relu', dropout=0.0)
        self.up7 = Block(mode='decoder', in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1,
                         batch_norm=True, activation='relu', dropout=0.0)
        self.up8 = Block(mode='decoder', in_channels=128, out_channels=out_channels, kernel_size=4, stride=2, padding=1,
                         batch_norm=False, activation='', dropout=0.0)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))
        u8 = self.up8(torch.cat([u7, d1], dim=1))

        return (torch.tanh(u8) + 1) / 2
