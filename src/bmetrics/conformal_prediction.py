import torch
import torch.nn as nn
import torch.optim as optim
from bmetrics.pretrained_models import load_experts
from fairchem.core.datasets import LmdbDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

def pinball_loss(y, pred, tau):
    residual = y - pred
    return torch.where(residual > 0, residual * tau, residual * (tau - 1))

weights_root = '/Users/averyhill/Github/BindingModelMetrics/models'
models = load_experts(model_names=['dimenetpp'], weights_root=weights_root, device='cpu')
model = models[0]

dataset = LmdbDataset({"src": 'data/oc20/train/raw'})
calibration_size = 150
dataset_size = len(dataset) - calibration_size
calibration_set, dataset = torch.utils.data.random_split(dataset, [calibration_size, dataset_size])
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=123)
qutrain_loader = DataLoader(train_set, batch_size=32, shuffle=False)
calibration_loader = DataLoader(calibration_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# Fine-tune the pretrained model to generate the upper and lower quantiles
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
tau = 0.95
for data in calibration_loader:
    # Forward pass
    pred = model(data)
    y = data.y_relaxed.unsqueeze(-1)
    loss = pinball_loss(y=y, pred=pred, tau=0.95)
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()