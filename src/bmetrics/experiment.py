def make_experiment():
    return [
        {"experts": ["dimenetpp"], "finetune": False, "moe": False},
        {"experts": ["schnet"], "finetune": False, "moe": False},
        {"experts": ["painn"], "finetune": False, "moe": False},
        {"experts": ["dimenetpp"], "finetune": True, "moe": False},
        {"experts": ["schnet"], "finetune": True, "moe": False},
        {"experts": ["painn"], "finetune": True, "moe": False},
        {"experts": ["dimenetpp", "schnet", "painn"], "finetune": False, "moe": False},
        {"experts": ["dimenetpp", "schnet", "painn"], "finetune": True, "moe": False},
        # {"experts": ["dimenetpp", "schnet", "painn"], "finetune": False, "moe": True},
        {"experts": ["dimenetpp", "schnet", "painn"], "finetune": True, "moe": True},
    ]
