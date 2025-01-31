# I want to import MSELoss
import torch
import torch.nn as nn


class ReducedQuantileLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.1,
    ) -> None:
        super(ReducedQuantileLoss, self).__init__()  # Call the parent class initializer
        self.quantiles = [alpha / 2, 0.5, 1 - alpha / 2]
        self.n_quantiles = len(self.quantiles)

    # Define the quantile loss
    def quantile_loss(self, y_true, y_pred, tau):
        error = y_true - y_pred
        loss = torch.mean(torch.max((tau - 1) * error, tau * error))
        return loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for tau in self.quantiles:
            loss = self.quantile_loss(target, pred[:, int(tau * self.n_quantiles)], tau)
            losses.append(loss)
        total_loss = torch.mean(torch.stack(losses))
        return total_loss


def get_calibration_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.unsqueeze(-1)
    score = torch.max(target - pred[:, 0], pred[:, 1] - target)
    return score
