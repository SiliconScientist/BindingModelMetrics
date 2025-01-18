# I want to import MSELoss
import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    def __init__(
        self,
        quantile: float,
    ) -> None:
        super(QuantileLoss, self).__init__()  # Call the parent class initializer
        self.quantile = quantile

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, reduction: bool = True
    ) -> torch.Tensor:
        target = target.unsqueeze(-1)
        losses = torch.max(
            self.quantile * (target - pred), (1 - self.quantile) * (pred - target)
        )
        if reduction:
            return losses.mean()
        return losses


def get_calibration_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.unsqueeze(-1)
    score = torch.max(target - pred[:, 0], pred[:, 1] - target)
    return score
