import toml
import torch
import torch.nn as nn
import torch.optim as optim
from fairchem.core.datasets import LmdbDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

import wandb
from bmetrics.config import Config
from bmetrics.models import GatingGCN, MixtureOfExperts
from bmetrics.pretrained_models import load_experts
from bmetrics.train import Trainer, evaluate


def main():
    config = Config(**toml.load("config.toml"))
    torch.manual_seed(config.random_seed)
    wandb_config = config.model.model_dump() | config.optimizer.model_dump()
    wandb.init(project="Binding Model Metrics", config=wandb_config)
    dataset = LmdbDataset({"src": str(config.paths.data)})
    if not config.subset_size == 0:
        dataset = Subset(dataset, indices=list(range(config.subset_size)))
    train, temp = train_test_split(
        dataset, test_size=0.1, random_state=config.random_seed
    )
    val, test = train_test_split(temp, test_size=0.5, random_state=config.random_seed)
    train_loader = DataLoader(
        train, batch_size=config.trainer.batch_size, shuffle=False
    )
    val_loader = DataLoader(val, batch_size=config.trainer.batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=config.trainer.batch_size, shuffle=False)
    experts = load_experts(
        model_names=config.model.names,
        models_path=config.paths.models,
        device=config.device,
    )
    gating_network = GatingGCN(
        **config.model.model_dump(exclude={"names"}), experts=experts
    )
    gating_network.to(config.device)
    model = MixtureOfExperts(
        experts=experts,
        gating_network=gating_network,
        device=config.device,
    )
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), **config.optimizer.model_dump())
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, **config.scheduler.model_dump()
    )
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    trainer.train()
    checkpoint = torch.load(config.paths.checkpoints)  # type: ignore
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test_loss = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=config.device,
    )
    print(f"Test Loss: {test_loss:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
