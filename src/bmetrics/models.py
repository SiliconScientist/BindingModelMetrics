import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from bmetrics.config import Config
from bmetrics.pretrained_models import load_experts


def set_hyperparameters(
    cfg: Config,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    max_epochs: int,
) -> Config:
    cfg.model.hidden_dim = hidden_dim
    cfg.model.num_layers = num_layers
    cfg.model.dropout = dropout
    cfg.optimizer.lr = lr
    cfg.optimizer.weight_decay = weight_decay
    cfg.trainer.max_epochs = max_epochs
    return cfg


class Ensemble(nn.Module):
    def __init__(self, experts: nn.ModuleList) -> None:
        super().__init__()
        self.experts = experts

    def forward(self, data):
        predictions = torch.stack([model(data) for model in self.experts], dim=1)
        return predictions.mean(dim=1).squeeze()


class GatingGCN(torch.nn.Module):
    def __init__(
        self,
        experts: nn.ModuleList,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        input_dim: int = 4,
    ):
        super().__init__()
        num_experts = len(experts)
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.lin = nn.Linear(hidden_dim, num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x = torch.cat([data.atomic_numbers.unsqueeze(1), data.pos], dim=-1)
        for conv in self.convs:
            x = conv(x, data.edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        x = self.lin(x)
        return F.softmax(x, dim=1).unsqueeze(2)


class MixtureOfExperts(nn.Module):
    def __init__(self, cfg, experts):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList(experts)
        self.gating_network = GatingGCN(**cfg.model.model_dump(), experts=experts).to(
            cfg.device
        )

    def forward(self, data):
        # Shape: [batch_size, num_experts, output_dim]
        predictions = torch.stack([model(data) for model in self.experts], dim=1)
        weights_matrix = self.gating_network(data)
        weighted_prediction = predictions * weights_matrix
        # Shape: [batch_size, output_dim]
        prediction = weighted_prediction.sum(dim=1).squeeze(-1)
        return prediction


def make_model(
    cfg: Config,
    expert_names: list[str],
    moe: bool,
) -> nn.Module:
    experts = load_experts(expert_names=expert_names, cfg=cfg)
    if moe:
        model = MixtureOfExperts(cfg, experts)
    else:
        model = Ensemble(experts)
    return model
