import toml
import torch
import torch.nn as nn
from torch_scatter import scatter
from fairchem.core.models.painn import PaiNN
from fairchem.core.models.schnet import SchNetWrap
from fairchem.core.models.dimenet_plus_plus import DimeNetPlusPlusWrapBackbone
from torch_geometric.nn import global_mean_pool

from bmetrics.config import Config


class DNPP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.emb_channels = 192
        self.output_dim = 3
        self.output_layer = nn.Linear(self.emb_channels, self.output_dim)

    def forward(self, data):
        x = self.model(data)
        # Shape: [num_nodes, emb_channels]
        x = scatter(x["edge_embedding"], x["edge_idx"], dim=0)
        # Shape: [batch_size, emb_channels]
        x = global_mean_pool(x, data.batch)
        # Shape: [batch_size, output_dim]
        x = self.output_layer(x)
        return x


class SN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data):
        return self.model(data)["energy"]


class PN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data):
        return self.model(data)["energy"].unsqueeze(-1)


MODEL_CLASSES = {
    "dimenetpp": (
        DimeNetPlusPlusWrapBackbone,
        DNPP,
        "dimenetpp_all.pt",
    ),
    "schnet": (SchNetWrap, SN, "schnet_all_large.pt"),
    "painn": (PaiNN, PN, "painn_all.pt"),
}


def load_model(name: str, config: Config) -> nn.Module:
    expert_config = toml.load("pretrained.toml")
    model_class, wrapper, weights_filename = MODEL_CLASSES[name]
    weights_path = config.paths.experts / weights_filename
    model = model_class(**expert_config[name])
    model = wrapper(model)
    weights = torch.load(
        weights_path, map_location=torch.device(config.device), weights_only=True
    )
    model.load_state_dict(weights["state_dict"], strict=False)
    model.to(config.device)
    return model


def load_experts(names: list, config: Config) -> nn.ModuleList:
    experts = nn.ModuleList([load_model(name=name, config=config) for name in names])
    return experts
