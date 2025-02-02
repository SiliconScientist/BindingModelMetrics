# I want to import MSELoss
import torch
import torch.nn as nn


class ReducedQuantileLoss(nn.Module):
    def __init__(self, quantiles) -> None:
        super(ReducedQuantileLoss, self).__init__()  # Call the parent class initializer
        self.quantiles = quantiles
        self.quantile_to_index = {
            quantile: idx for idx, quantile in enumerate(quantiles)
        }

    def quantile_loss(self, y_true, y_pred, tau):
        error = y_true - y_pred
        loss = torch.mean(torch.max((tau - 1) * error, tau * error))
        return loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for tau in self.quantiles:
            idx = self.quantile_to_index[tau]
            loss = self.quantile_loss(target, pred[:, idx], tau)
            losses.append(loss)
        total_loss = torch.mean(torch.stack(losses))
        return total_loss


def get_calibration_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.unsqueeze(-1)
    score = torch.max(target - pred[:, 0], pred[:, 1] - target)
    return score
