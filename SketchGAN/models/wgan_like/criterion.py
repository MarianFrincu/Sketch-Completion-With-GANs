import torch
import torch.nn as nn


class WGeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake_scores):
        loss = -torch.mean(fake_scores)
        return loss


class WCriticLoss(nn.Module):
    def __init__(self, lambda_gp=10):
        super().__init__()
        self.lambda_gp = lambda_gp

    def forward(self, real_scores, fake_scores, penalty=0):
        loss = torch.mean(fake_scores) - torch.mean(real_scores) + self.lambda_gp * penalty
        return loss
