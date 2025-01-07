import torch


@torch.no_grad()
def evaluate(model, criterion, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    loss = 0.0
    for data in dataloader:
        data = data.to(device)
        pred = model(data)
        loss = criterion(pred, data.energy)
        loss += loss.item()
    loss /= len(dataloader)
    return loss
