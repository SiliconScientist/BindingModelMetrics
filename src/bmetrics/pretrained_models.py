from pathlib import Path

import toml
import torch
from fairchem.core.models.dimenet_plus_plus import DimeNetPlusPlus, DimeNetPlusPlusWrap
import torch.nn as nn
from fairchem.core.models.equiformer_v2.equiformer_v2 import (
    EquiformerV2Backbone,
    EquiformerV2EnergyHead,
)
from fairchem.core.models.painn import PaiNN
from fairchem.core.models.schnet import SchNet, SchNetWrap


def set_up_model(
    model_class,
    model_arguments: dict,
    weights_path: str,
    device: str,
) -> torch.nn.Module:
    model = model_class(**model_arguments).to(device)
    weights = torch.load(
        weights_path, map_location=torch.device(device), weights_only=True
    )
    model.load_state_dict(weights["state_dict"], strict=False)
    return model


def load_experts(model_names: list, models_path: Path, device: str) -> list[nn.Module]:
    config = toml.load("pretrained.toml")
    experts = []
    if "dimenetpp" in model_names:
        weights_path = f"{models_path}/dimenetpp_all.pt"
        model = set_up_model(
            model_class=DimeNetPlusPlusWrap,
            model_arguments=config["dimenetpp"],
            weights_path=weights_path,
            device=device,
        )
        experts.append(model)

    if "schnet" in model_names:
        weights_path = f"{models_path}/schnet_all_large.pt"
        model = set_up_model(
            model_class=SchNetWrap,
            model_arguments=config["schnet"],
            weights_path=weights_path,
            device=device,
        )
        experts.append(model)
    if "painn" in model_names:
        weights_path = f"{models_path}/painn/painn_all.pt"
        model = set_up_model(
            model_class=PaiNN,
            model_arguments=config["painn"],
            weights_path=weights_path,
            device=device,
        )
        experts.append(model)
    if "equiformerv2" in model_names:
        weights_path = f"{models_path}/eq2_153M_ec4_allmd.pt"
        model = set_up_model(
            model_class=EquiformerV2Backbone,
            model_arguments=config["equiformerv2"],
            weights_path=weights_path,
            device=device,
        )
        experts.append(model)
    return experts


def get_expert_output(data, model):
    """
    Interface between MixtureOfExperts class and expert models.

    Returns the expert predictions with shape [batch_size, output_dim]
    """
    match model:
        case DimeNetPlusPlus():
            prediction = model(data)["energy"]
            return prediction
        case SchNet():
            prediction = model(data)["energy"]
            return prediction
        case PaiNN():  # type: ignore
            prediction = model(data)["energy"]
            return prediction.unsqueeze(1)
        case EquiformerV2Backbone():  # type: ignore
            energy_head = EquiformerV2EnergyHead(model)
            emb = model(data)
            prediction = energy_head(data=data, emb=emb)["energy"].unsqueeze(1)
            return prediction
        case _:
            raise ValueError(f"Model '{model}' not recognized")
