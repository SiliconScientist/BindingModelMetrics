import os
import toml
import torch
from fairchem.core.models.dimenet_plus_plus import DimeNetPlusPlusWrap
import torch.nn as nn
from fairchem.core.models.painn import PaiNN
from fairchem.core.models.schnet import SchNetWrap


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


def load_model(name: str, device: str) -> nn.Module:
    config = toml.load("pretrained.toml")
    model_class, wrapper, weights_filename = MODEL_CLASSES[name]
    weights_path = os.path.join("experts", weights_filename)
    model = model_class(**config[name])
    model = wrapper(model)
    weights = torch.load(
        weights_path, map_location=torch.device(device), weights_only=True
    )
    model.load_state_dict(weights["state_dict"], strict=False)
    model.to(device)
    return model


def load_experts(names: list, device: str) -> list[nn.Module]:
    experts = [load_model(name=name, device=device) for name in names]
    return experts
