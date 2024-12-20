import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GatingGCN(torch.nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_channels: int):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_experts)

    def forward(self, data):
        x = torch.cat([data.atomic_numbers.unsqueeze(1), data.pos], dim=-1)
        x = self.conv1(x, data.edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, data.edge_index)
        x = self.lin(x)
        x = global_mean_pool(x, data.batch)
        return F.log_softmax(x, dim=1)


class MixtureOfExperts(nn.Module):
    def __init__(self, trained_experts, gating_network, device):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList(trained_experts)
        self.gating_network = gating_network.to(device)

    def forward(self, data):
        # Matrix shape: [batch_size, num_experts, output_dim]
        prediction_matrix = torch.stack([expert(data)['energy'] for expert in self.experts], dim=1)
        weights_matrix = self.gating_network(data).unsqueeze(2)
        weighted_prediction_matrix = prediction_matrix * weights_matrix
        # Shape: [batch_size, output_dim]
        prediction = weighted_prediction_matrix.sum(dim=1)
        return prediction
    
class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)