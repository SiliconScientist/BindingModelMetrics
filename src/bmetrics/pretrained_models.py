import torch
from torch_geometric.nn.models.dimenet import DimeNetPlusPlus
from torch_geometric.nn.models.schnet import SchNet

def set_up_model(model_class, model_arguments: dict, weights_path: str, device: str, freeze_parameters=False,) -> torch.nn.Module:
    model = model_class(**model_arguments)
    weights = torch.load(weights_path, map_location=torch.device(device), weights_only=True)
    model.load_state_dict(weights["state_dict"], strict=False)
    if freeze_parameters:
        for param in model.parameters():
            param.requires_grad = False
    return model

def load_experts(models_names: list, weights_root: str, device: str) -> list:
    experts = []
    if "dimenetpp" in models_names:
        model_arguments = {
        'hidden_channels': 192,
        'out_emb_channels': 192,
        'int_emb_size': 64,
        'out_channels': 1,
        'basis_emb_size': 8,
        'num_blocks': 3,
        'cutoff': 6.0,
        'num_radial': 6,
        'num_spherical': 7,
        'num_before_skip': 1,
        'num_after_skip': 2,
        'num_output_layers': 3,
        }
        weights_path = f"{weights_root}/dimenetpp_all.pt"
        model = set_up_model(model_class=DimeNetPlusPlus, model_arguments=model_arguments, weights_path=weights_path, device=device)
        experts.append(model)

    if "schnet" in models_names:
        model_arguments = {
        'hidden_channels': 1024,
        'num_filters': 256,
        'num_interactions': 5,
        'num_gaussians': 200,
        'cutoff': 6.0,
        }
        weights_path = f"{weights_root}/schnet_all_large.pt"
        model = set_up_model(model_class=SchNet, model_arguments=model_arguments, weights_path=weights_path, device=device)
        experts.append(model)
    return experts