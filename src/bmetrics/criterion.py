# I want to import MSELoss
import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.1,
    ) -> None:
        super(QuantileLoss, self).__init__()  # Call the parent class initializer
        self.quantiles = [alpha / 2, 1 - alpha / 2]
        self.n_quantiles = len(self.quantiles)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for tau in self.quantiles:
            error = target - pred[:, int(tau * self.n_quantiles)].unsqueeze(1)
            loss = torch.mean(torch.max((tau - 1) * error, tau * error))
            losses.append(loss)
        total_loss = torch.mean(torch.stack(losses))
        return total_loss


def get_calibration_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.unsqueeze(-1)
    score = torch.max(target - pred[:, 0], pred[:, 1] - target)
    return score
