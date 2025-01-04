from pathlib import Path
import torch
from fairchem.core.models.dimenet_plus_plus import DimeNetPlusPlusWrap, DimeNetPlusPlus
from fairchem.core.models.schnet import SchNetWrap, SchNet
from fairchem.core.models.painn import PaiNN
from fairchem.core.models.equiformer_v2.equiformer_v2 import (
    EquiformerV2Backbone,
    EquiformerV2EnergyHead,
)


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


def load_experts(model_names: list, models_path: Path, device: str) -> list:
    experts = []
    if "dimenetpp" in model_names:
        model_arguments = {
            "hidden_channels": 192,
            "out_emb_channels": 192,
            "num_blocks": 3,
            "cutoff": 6.0,
            "num_radial": 6,
            "num_spherical": 7,
            "num_before_skip": 1,
            "num_after_skip": 2,
            "num_output_layers": 3,
            "use_pbc": True,
            "otf_graph": True,
        }
        weights_path = f"{models_path}/dimenetpp_all.pt"
        model = set_up_model(
            model_class=DimeNetPlusPlusWrap,
            model_arguments=model_arguments,
            weights_path=weights_path,
            device=device,
        )
        experts.append(model)

    if "schnet" in model_names:
        model_arguments = {
            "hidden_channels": 1024,
            "num_filters": 256,
            "num_interactions": 5,
            "num_gaussians": 200,
            "cutoff": 6.0,
            "use_pbc": True,
            "otf_graph": True,
        }
        weights_path = f"{models_path}/schnet_all_large.pt"
        model = set_up_model(
            model_class=SchNetWrap,
            model_arguments=model_arguments,
            weights_path=weights_path,
            device=device,
        )
        experts.append(model)
    if "painn" in model_names:
        model_arguments = {
            "hidden_channels": 512,
            "num_layers": 6,
            "num_rbf": 128,
            "cutoff": 12.0,
            "max_neighbors": 50,
            "scale_file": f"{models_path}/painn/painn_nb6_scaling_factors.pt",
            "regress_forces": True,
            "direct_forces": True,
            "use_pbc": True,
        }
        weights_path = f"{models_path}/painn/painn_all.pt"
        model = set_up_model(
            model_class=PaiNN,
            model_arguments=model_arguments,
            weights_path=weights_path,
            device=device,
        )
        experts.append(model)
    if "equiformerv2" in model_names:
        model_arguments = {
            "use_pbc": True,
            "regress_forces": True,
            "otf_graph": True,
            "max_neighbors": 20,
            "max_radius": 12.0,
            "max_num_elements": 90,
            "num_layers": 20,
            "sphere_channels": 128,
            "attn_hidden_channels": 64,  # [64, 96] This determines the hidden size of message passing. Do not necessarily use 96.
            "num_heads": 8,
            "attn_alpha_channels": 64,  # Not used when `use_s2_act_attn` is True.
            "attn_value_channels": 16,
            "ffn_hidden_channels": 128,
            "norm_type": "layer_norm_sh",  # ['rms_norm_sh', 'layer_norm', 'layer_norm_sh']
            "lmax_list": [6],
            "mmax_list": [3],
            "grid_resolution": 18,  # [18, 16, 14, None] For `None`, simply comment this line.
            "num_sphere_samples": 128,
            "edge_channels": 128,
            "use_atom_edge_embedding": True,
            "distance_function": "gaussian",
            "num_distance_basis": 512,  # not used
            "attn_activation": "silu",
            "use_s2_act_attn": False,  # [False, True] Switch between attention after S2 activation or the original EquiformerV1 attention.
            "ffn_activation": "silu",  # ['silu', 'swiglu']
            "use_gate_act": False,  # [False, True] Switch between gate activation and S2 activation
            "use_grid_mlp": True,  # [False, True] If `True`, use projecting to grids and performing MLPs for FFNs.
            "alpha_drop": 0.1,  # [0.0, 0.1]
            "drop_path_rate": 0.1,  # [0.0, 0.05]
            "proj_drop": 0.0,
            "weight_init": "uniform",  # ['uniform', 'normal']
        }
        weights_path = f"{models_path}/eq2_153M_ec4_allmd.pt"
        model = set_up_model(
            model_class=EquiformerV2Backbone,
            model_arguments=model_arguments,
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
    if isinstance(model, DimeNetPlusPlus):
        prediction = model(data)["energy"]
        return prediction
    elif isinstance(model, SchNet):
        prediction = model(data)["energy"]
        return prediction
    elif isinstance(model, PaiNN):  # type: ignore
        prediction = model(data)["energy"]  # type: ignore
        return prediction.unsqueeze(1)
    elif isinstance(model, EquiformerV2Backbone):  # type: ignore
        energy_head = EquiformerV2EnergyHead(model)
        emb = model(data)
        prediction = energy_head(data=data, emb=emb)["energy"].unsqueeze(1)
        return prediction
