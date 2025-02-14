import toml
import torch
import torch.nn as nn
from fairchem.core.models.dimenet_plus_plus import DimeNetPlusPlusWrap
from fairchem.core.models.painn import PaiNN
from fairchem.core.models.schnet import SchNetWrap
import torch._dynamo

from bmetrics.config import Config

torch._dynamo.config.suppress_errors = True


class DNPP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data):
        return self.model(data)["energy"]


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
        return self.model(data)["energy"].unsqueeze(1)


MODEL_CLASSES = {
    "dimenetpp": (DimeNetPlusPlusWrap, DNPP, "dimenetpp_all.pt"),
    "schnet": (SchNetWrap, SN, "schnet_all_large.pt"),
    "painn": (PaiNN, PN, "painn_all.pt"),
}


def load_model(name: str, cfg: Config) -> nn.Module:
    expert_cfg = toml.load("pretrained.toml")
    model_class, wrapper, weights_filename = MODEL_CLASSES[name]
    weights_path = cfg.paths.experts / weights_filename
    model = model_class(**expert_cfg[name])
    model = wrapper(model)
    weights = torch.load(
        weights_path, map_location=torch.device(cfg.device), weights_only=True
    )
    model.load_state_dict(weights["state_dict"], strict=False)
    model.to(cfg.device)
    model = torch.compile(model)
    return model  # type: ignore


def load_experts(expert_names: list[str], cfg: Config) -> nn.ModuleList:
    experts = nn.ModuleList([load_model(name=name, cfg=cfg) for name in expert_names])
    return experts
