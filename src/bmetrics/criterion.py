# I want to import MSELoss
import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    def __init__(
        self,
        quantiles: list[float] = [0.05, 0.95],
    ) -> None:
        super(QuantileLoss, self).__init__()  # Call the parent class initializer
        self.quantiles = torch.tensor(quantiles)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, reduction: bool = True
    ) -> torch.Tensor:
        target = torch.stack([target], dim=1)
        residual = target - pred
        losses = torch.max((self.quantiles - 1) * residual, self.quantiles * residual)
        if reduction:
            return losses.mean()
        return losses


def get_calibration_score(losses: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = torch.stack([target], dim=1)
    upper_bound = torch.max(losses, dim=1).values
    lower_bound = torch.min(losses, dim=1).values
    score = torch.max(target - upper_bound, lower_bound - target)
    return score
