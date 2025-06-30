import torch
from torch import nn


class GeneratorLoss(nn.Module):
    def __init__(self, lambda1=100, lambda2=0.5):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.bce = nn.BCELoss()
        self.l1 = nn.L1Loss()

    def forward(self, original, generated, fake_pred, classifier_loss):
        fake_target = torch.full_like(fake_pred, 1)
        bce = self.bce(fake_pred, fake_target)
        l1 = self.l1(original, generated)
        loss = (bce + self.lambda1 * l1 + self.lambda2 * classifier_loss)
        return loss


class DiscriminatorLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, real_pred, fake_pred):
        real_target = torch.full_like(real_pred, 1)
        fake_target = torch.full_like(fake_pred, 0)
        real_target_loss = self.bce(real_pred, real_target)
        fake_target_loss = self.bce(fake_pred, fake_target)
        loss = real_target_loss + fake_target_loss
        return loss
