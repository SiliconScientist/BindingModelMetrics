import torch
from fairchem.core.models.dimenet_plus_plus import DimeNetPlusPlusWrap
from fairchem.core.models.schnet import SchNetWrap
from fairchem.core.models.painn import PaiNN


def set_up_model(model_class, model_arguments: dict, weights_path: str, device: str, freeze_parameters=False,) -> torch.nn.Module:
    model = model_class(**model_arguments).to(device)
    weights = torch.load(weights_path, map_location=torch.device(device), weights_only=True)
    model.load_state_dict(weights["state_dict"], strict=False)
    if freeze_parameters:
        for param in model.parameters():
            param.requires_grad = False
    return model

def load_experts(model_names: list, weights_root: str, device: str) -> list:
    experts = []
    if "dimenetpp" in model_names:
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
        'regress_forces': True,
        'use_pbc': True,
        }
        weights_path = f"{weights_root}/dimenetpp_all.pt"
        model = set_up_model(model_class=DimeNetPlusPlusWrap, model_arguments=model_arguments, weights_path=weights_path, device=device)
        experts.append(model)

    if "schnet" in model_names:
        model_arguments = {
        'hidden_channels': 1024,
        'num_filters': 256,
        'num_interactions': 5,
        'num_gaussians': 200,
        'cutoff': 6.0,
        'use_pbc': True,
        }
        weights_path = f"{weights_root}/schnet_all_large.pt"
        model = set_up_model(model_class=SchNetWrap, model_arguments=model_arguments, weights_path=weights_path, device=device)
        experts.append(model)

    if 'painn' in model_names:
        model_arguments = {
            'hidden_channels': 512,
            'num_layers': 6,
            'num_rbf': 128,
            'cutoff': 12.0,
            'max_neighbors': 50,
            'scale_file': 'models/painn/painn_nb6_scaling_factors.pt',
            'regress_forces': True,
            'direct_forces': True,
            'use_pbc': True,
        }
        weights_path = f"{weights_root}/painn/painn_all.pt"
        model = set_up_model(model_class=PaiNN, model_arguments=model_arguments, weights_path=weights_path, device=device)
        experts.append(model)
    return experts