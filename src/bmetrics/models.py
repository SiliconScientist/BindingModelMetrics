import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from bmetrics.pretrained_models import get_expert_output


class GatingGCN(torch.nn.Module):
    def __init__(
        self, input_dim: int, num_experts: int, hidden_dim: int, num_layers: int
    ):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.lin = nn.Linear(hidden_dim, num_experts)

    def forward(self, data):
        x = torch.cat([data.atomic_numbers.unsqueeze(1), data.pos], dim=-1)
        for conv in self.convs:
            x = conv(x, data.edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = self.lin(x)
        return F.softmax(x, dim=1)


class MixtureOfExperts(nn.Module):
    def __init__(self, trained_experts, gating_network, device):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList(trained_experts)
        self.gating_network = gating_network.to(device)

    def forward(self, data):
        # Shape: [batch_size, num_experts, output_dim]
        prediction_matrix = torch.stack(
            [get_expert_output(data=data, model=expert) for expert in self.experts],
            dim=1,
        )
        weights_matrix = self.gating_network(data).unsqueeze(2)
        weighted_prediction_matrix = prediction_matrix * weights_matrix
        # Shape: [batch_size, output_dim]
        prediction = weighted_prediction_matrix.sum(dim=(1, 2))
        return prediction
