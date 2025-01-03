import toml
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import wandb
from bmetrics.pretrained_models import load_experts
from bmetrics.models import GatingGCN, MixtureOfExperts
from bmetrics.config import Config
from sklearn.model_selection import train_test_split
from fairchem.core.datasets import LmdbDataset
from torch.utils.data import Subset


def main():
    config = Config(**toml.load("config.toml"))
    wandb.init(project="Binding Model Metrics", 
               config={'hidden_dim': config.hidden_dim,
                       'batch_size': config.batch_size,
                       'lr': config.lr,
                       'momentum': config.momentum,
                       'nesterov': config.nesterov,
                       'gamma': config.gamma,
                       'max_epochs': config.max_epochs,
                       }
                )
    dataset = LmdbDataset({"src": config.data_root})
    if not config.subset_size == 0:
        dataset = Subset(dataset, indices=list(range(config.subset_size)))
    train, temp = train_test_split(dataset, test_size=0.3, random_state=config.random_seed)
    val, test = train_test_split(temp, test_size=0.5, random_state=config.random_seed)
    train_dataloader = DataLoader(train, batch_size=config.batch_size, shuffle=False)
    val_dataloader = DataLoader(val, batch_size=config.batch_size, shuffle=False)
    test_dataloader = DataLoader(test, batch_size=config.batch_size, shuffle=False)
    trained_experts = load_experts(model_names=config.model_names, models_root=config.models_root, device=config.device)
    num_experts = len(trained_experts)
    gating_network = GatingGCN(input_dim=config.input_dim, num_experts=num_experts, hidden_dim=config.hidden_dim, num_layers=config.num_layers).to(config.device)
    model = MixtureOfExperts(trained_experts=trained_experts, gating_network=gating_network, device=config.device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, nesterov=config.nesterov)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
    for epoch in range(config.max_epochs):
        model.train()
        train_loss = 0.0
        for data in train_dataloader:
            data = data.to(config.device)
            pred = model(data)
            loss = criterion(pred, data.energy.unsqueeze(1))
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient computation
            for data in val_dataloader:
                data = data.to(config.device)
                pred = model(data)
                loss = criterion(pred, data.energy.unsqueeze(1))
                val_loss += loss.item()
        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
        print(f"Epoch {epoch+1}/{config.max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in test_dataloader:
            data = data.to(config.device)
            pred = model(data)
            loss = criterion(pred, data.energy.unsqueeze(1))
            total_loss += loss.item()
    average_loss = total_loss / len(test_dataloader)
    print(f"Test Loss: {average_loss}")
    wandb.finish()


if __name__ == "__main__":
    main()
