import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader

import wandb
from bmetrics.config import Config


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


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: nn.MSELoss,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_val_loss = float("inf")
        self.config = config

    def train(self) -> None:
        for epoch in range(self.config.max_epochs):
            self.model.train()
            train_loss = 0.0
            for data in self.train_loader:
                data = data.to(self.config.device)
                self.optimizer.zero_grad()
                pred = self.model(data)
                loss = self.criterion(pred, data.energy)
                train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            self.scheduler.step()
            train_loss /= len(self.train_loader)
            val_loss = evaluate(
                model=self.model,
                dataloader=self.val_loader,
                criterion=self.criterion,
                device=self.config.device,
            )
            wandb.log(
                {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
            )
            print(
                f"Epoch {epoch+1}/{self.config.max_epochs}, "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": val_loss,
                    },
                    self.config.paths.checkpoints,
                )
                print(f"New best model saved to {self.config.paths.checkpoints}")
