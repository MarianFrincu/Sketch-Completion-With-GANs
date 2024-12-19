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
        fake_target = torch.ones_like(fake_pred)
        bce = self.bce(fake_pred, fake_target)
        l1 = self.l1(original, generated)
        loss = (bce + self.lambda1 * l1 + self.lambda2 * classifier_loss)
        return loss


class DiscriminatorLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.bce1 = nn.BCELoss()
        self.bce2 = nn.BCELoss()

    def forward(self, real_pred, fake_pred):
        real_target = torch.ones_like(real_pred)
        fake_target = torch.zeros_like(fake_pred)
        real_target_loss = self.bce1(real_pred, real_target)
        fake_target_loss = self.bce2(fake_pred, fake_target)
        loss = real_target_loss + fake_target_loss
        return loss
