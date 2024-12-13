import torch
from bmetrics.pretrained_models import load_experts

def pinball_loss(y, pred, tau):
    residual = y - pred
    return torch.where(residual > 0, residual * tau, residual * (tau - 1))

