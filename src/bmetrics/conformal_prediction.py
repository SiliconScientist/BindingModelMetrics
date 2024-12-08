import torch
from bmetrics.pretrained_models import load_experts
from fairchem.core.datasets import LmdbDataset
from torch_geometric.loader import DataLoader

weights_root = '/Users/averyhill/Github/BindingModelMetrics/models'
models = load_experts(model_names=['dimenetpp'], weights_root=weights_root, device='cpu')
model = models[0]

dataset = LmdbDataset({"src": 'data/oc20/train/raw'})
calibration_size = 150
test_size = len(dataset) - calibration_size
calibration_set, test_set = torch.utils.data.random_split(dataset, [calibration_size, test_size])
calibration_loader = DataLoader(calibration_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

with torch.no_grad():
    for data in calibration_loader:
        print(data)
        predictions = model(data)  # Forward pass
        residual = data.y_relaxed.unsqueeze(-1) - predictions
        loss = torch.where(residual >= 0
            predictions, data.y_relaxed.unsqueeze(-1))
        total_loss += loss.item()
    average_loss = total_loss / len(test_loader)
    print(f"Test Loss: {average_loss}")