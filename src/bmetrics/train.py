import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader

import wandb
from bmetrics.config import Config
from bmetrics.dataset import DataloaderSplits


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: nn.MSELoss,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Config,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.best_val_loss = float("inf")
        self.config = config

    def train_step(self, data):
        data = data.to(self.config.device)
        self.optimizer.zero_grad()
        pred = self.model(data)
        loss = self.criterion(pred, data.energy)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    def train(self) -> None:
        if self.config.fast_dev_run:
            self.config.trainer.max_epochs = 1
        for epoch in range(self.config.trainer.max_epochs):
            self.model.train()
            train_loss = 0.0
            for data in self.train_loader:
                train_loss += self.train_step(data)
            self.scheduler.step()
            train_loss /= len(self.train_loader)
            val_loss = self.validate()
            if self.config.log:
                wandb.log(
                    {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
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
        weights = torch.load(self.config.paths.checkpoints)
        self.model.load_state_dict(weights["model_state_dict"])

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()  # Set the model to evaluation mode
        loss = 0.0
        for data in dataloader:
            data = data.to(self.config.device)
            pred = self.model(data)
            loss = self.criterion(pred, data.energy)
            loss += loss.item()
        loss /= len(dataloader)
        return loss

    def validate(self) -> float:
        return self.evaluate(self.val_loader)

    def test(self) -> float:
        return self.evaluate(self.test_loader)


def make_trainer(
    config: Config, dataloaders: DataloaderSplits, model: nn.Module
) -> Trainer:
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
        train_loader=dataloaders.train,
        val_loader=dataloaders.val,
        test_loader=dataloaders.test,
        config=config,
    )
    return trainer
