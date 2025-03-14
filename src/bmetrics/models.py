import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from bmetrics.config import Config
from bmetrics.pretrained_models import load_experts


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
        input_dim: int,
        experts: nn.ModuleList,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
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
    def __init__(self, experts, gating_network, device):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList(experts)
        self.gating_network = gating_network.to(device)

    def forward(self, data):
        # Shape: [batch_size, num_experts, output_dim]
        predictions = torch.stack([model(data) for model in self.experts], dim=1)
        weights_matrix = self.gating_network(data)
        weighted_prediction = predictions * weights_matrix
        # Shape: [batch_size, output_dim]
        prediction = weighted_prediction.sum(dim=1).squeeze(-1)
        return prediction


def make_moe(config: Config, experts: nn.ModuleList):
    gating_network = GatingGCN(
        **config.model.model_dump(exclude={"names"}), experts=experts
    )
    gating_network.to(config.device)
    model = MixtureOfExperts(
        experts=experts,
        gating_network=gating_network,
        device=config.device,
    )
    return model


def make_model(config: Config, expert_names: list[str], moe: bool) -> nn.Module:
    experts = load_experts(names=expert_names, config=config)
    if moe:
        model = make_moe(config, experts)
    else:
        model = Ensemble(experts)
    return model
