import torch
import torch.nn as nn
from torch_geometric.data import InMemoryDataset
from torch.utils.data import Dataset, Subset
from fairchem.core.datasets import LmdbDataset
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from bmetrics.pretrained_models import get_expert_output




class GatingGCN(torch.nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_channels: int):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_experts)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.lin(x)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)


class MyDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(MyDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_paths(self):
        return ['data.lmdb']
    
    @property
    def processed_paths(self):
        return ['data_processed.pt']
    
    def download(self):
        pass

    def process(self):
        self.data = LmdbDataset({"src": self.raw_dir})
        # data_list = []
        # for data in self.data:
        #     x = data.atomic_numbers
        #     pos = data.pos_relaxed
        #     edge_index = data.edge_index
        #     y = data.y_relaxed

        #     # Create a PyTorch Geometric Data object
        #     data = Data(x=x, pos=pos, edge_index=edge_index, y=y)
        #     data_list.append(data)
        # data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])

    
    def __getitem__(self, idx: int):
        data = self.get(idx)  # PyG Data object
        return data


# Mixture of Experts Model
class MixtureOfExperts(nn.Module):
    def __init__(self, trained_experts, gating_network):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList(trained_experts)
        self.gating_network = gating_network

    def forward(self, data):
        expert_outputs = [get_expert_output(expert, data) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=2) # shape: (batch_size, num_experts, output_dim)
        x = torch.cat([data.atomic_numbers.unsqueeze(1), data.pos], dim=-1)
        weights = self.gating_network(x, data.edge_index, data.batch)
        weights = weights.unsqueeze(1).expand_as(expert_outputs)
        return torch.sum(expert_outputs * weights, dim=2)
