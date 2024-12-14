import toml
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from bmetrics.pretrained_models import load_experts
from bmetrics.models import GatingGCN, MixtureOfExperts, QuantileRegression
from bmetrics.config import Config
from sklearn.model_selection import train_test_split
from fairchem.core.datasets import LmdbDataset
from torch.utils.data import Subset
from pytorch_forecasting.metrics.quantile import QuantileLoss


config = Config(**toml.load("config.toml"))
dataset = LmdbDataset({"src": config.data_root})
if not config.subset_size == 0:
    dataset = Subset(dataset, indices=list(range(config.subset_size)))
train, test = train_test_split(dataset, test_size=0.2, random_state=config.random_seed)
train_dataloader = DataLoader(train, batch_size=config.batch_size, shuffle=False, drop_last=True)
test_dataloader = DataLoader(test, batch_size=config.batch_size, shuffle=False, drop_last=True)
trained_experts = load_experts(model_names=config.model_names, weights_root=config.weights_root, device=config.device)
base_model = trained_experts[0]
quantiles = [0.05, 0.95]
num_quantiles = len(quantiles)
model = QuantileRegression(base_model=base_model, batch_size=config.batch_size, num_quantiles=num_quantiles)
# num_experts = len(trained_experts)
# gating_network = GatingGCN(input_dim=config.input_dim, num_experts=num_experts, hidden_channels=config.hidden_channels).to(config.device)
# model = MixtureOfExperts(trained_experts=trained_experts, gating_network=gating_network, device=config.device)
criterion = QuantileLoss(quantiles=quantiles)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
for epoch in range(config.num_epochs):
    for data in train_dataloader:
        # Forward pass
        data = data.to(config.device)
        pred = model(data)
        loss = criterion(pred, data.energy.unsqueeze(-1))
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{config.num_epochs}], Loss: {loss.item():.4f}")
model.eval()
total_loss = 0.0
all_preds = []
all_targets = []
with torch.no_grad():
    for data in test_dataloader:
        data = data.to(config.device)
        pred = model(data)
        loss = criterion(pred, data.energy.unsqueeze(-1))
        total_loss += loss.item()
        all_preds.append(pred)
        all_targets.append(data.energy)
average_loss = total_loss / len(test_dataloader)
print(f"Test Loss: {average_loss}")