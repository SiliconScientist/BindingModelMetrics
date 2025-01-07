import os

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
from bmetrics.train import evaluate


def main():
    config = Config(**toml.load("config.toml"))
    wandb.init(
        project="Binding Model Metrics",
        config={
            "hidden_dim": config.hidden_dim,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "momentum": config.momentum,
            "nesterov": config.nesterov,
            "gamma": config.gamma,
            "max_epochs": config.max_epochs,
        },
    )
    dataset = LmdbDataset({"src": str(config.paths.data)})
    if not config.subset_size == 0:
        dataset = Subset(dataset, indices=list(range(config.subset_size)))
    train, temp = train_test_split(
        dataset, test_size=0.1, random_state=config.random_seed
    )
    val, test = train_test_split(temp, test_size=0.5, random_state=config.random_seed)
    train_dataloader = DataLoader(train, batch_size=config.batch_size, shuffle=False)
    val_dataloader = DataLoader(val, batch_size=config.batch_size, shuffle=False)
    test_dataloader = DataLoader(test, batch_size=config.batch_size, shuffle=False)
    trained_experts = load_experts(
        model_names=config.model_names,
        models_path=config.paths.models,
        device=config.device,
    )
    num_experts = len(trained_experts)
    gating_network = GatingGCN(
        input_dim=config.input_dim,
        num_experts=num_experts,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    )
    gating_network.to(config.device)
    model = MixtureOfExperts(
        trained_experts=trained_experts,
        gating_network=gating_network,
        device=config.device,
    )
    criterion = nn.MSELoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        nesterov=config.nesterov,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
    best_val_loss = float("inf")
    best_checkpoint_path = None
    os.makedirs(config.paths.checkpoints, exist_ok=True)
    for epoch in range(config.max_epochs):
        model.train()
        train_loss = 0.0
        for data in train_dataloader:
            data = data.to(config.device)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, data.energy)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        train_loss /= len(train_dataloader)
        val_loss = evaluate(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device=config.device,
        )
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"Epoch {epoch+1}/{config.max_epochs}, "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
        if val_loss < best_val_loss:
            if best_checkpoint_path:
                os.remove(best_checkpoint_path)
            best_val_loss = val_loss
            best_checkpoint_path = (
                f"{config.paths.checkpoints}/best_model_epoch_{epoch + 1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                },
                best_checkpoint_path,
            )
            print(f"New best model saved to {best_checkpoint_path}")
    checkpoint = torch.load(best_checkpoint_path)  # type: ignore
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test_loss = evaluate(
        model=model,
        dataloader=test_dataloader,
        criterion=criterion,
        device=config.device,
    )
    print(f"Test Loss: {test_loss:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
