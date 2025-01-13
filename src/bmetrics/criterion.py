# I want to import MSELoss
import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    def __init__(self, quantiles: list[float]):
        self.quantiles = torch.Tensor(quantiles)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        residual = target - pred
        loss = torch.max((self.quantiles - 1) * residual, self.quantiles * residual)
        quantile_loss = torch.mean(torch.sum(loss, dim=1))
        return quantile_loss
