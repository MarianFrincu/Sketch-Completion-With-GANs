import torch
from torch import nn


class GeneratorLoss(nn.Module):
    def __init__(self, lambda1=100, lambda2=0.5):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, original, generated, fake_pred, classifier_loss):
        fake_target = torch.ones_like(fake_pred)
        loss = (self.bce(fake_pred, fake_target) + self.lambda1 * self.l1(original, generated)
                + self.lambda2 * classifier_loss)
        return loss


class DiscriminatorLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, fake_pred, real_pred):
        fake_target = torch.zeros_like(fake_pred)
        real_target = torch.ones_like(real_pred)
        fake_loss = self.bce(fake_pred, fake_target)
        real_loss = self.bce(real_pred, real_target)
        loss = (fake_loss + real_loss) / 2
        return loss
