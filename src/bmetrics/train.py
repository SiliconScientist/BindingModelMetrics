import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader

import wandb
from bmetrics.config import Config


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: nn.MSELoss,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        cfg: Config,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_train_loss = float("inf")
        self.best_val_loss = float("inf")
        self.cfg = cfg

    def train_step(self, data):
        data = data.to(self.cfg.device)
        self.optimizer.zero_grad()
        pred = self.model(data)
        loss = self.criterion(pred, data.energy)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    def train(self, train_loader, eval_loader: DataLoader | None = None) -> None:
        if self.cfg.fast_dev_run:
            self.cfg.trainer.max_epochs = 1
        for epoch in range(self.cfg.trainer.max_epochs):
            self.model.train()
            train_loss = 0.0
            for data in train_loader:
                train_loss += self.train_step(data)
            self.scheduler.step()
            train_loss /= len(train_loader)
            if eval_loader:
                val_loss = self.evaluate(eval_loader)
                self.best_train_loss = train_loss
                self.best_val_loss = val_loss
                if self.cfg.log:
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                        }
                    )

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        loss = 0.0
        for data in dataloader:
            data = data.to(self.cfg.device)
            pred = self.model(data)
            loss = self.criterion(pred, data.energy)
            loss += loss.item()
        loss /= len(dataloader)
        return loss

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )


def make_trainer(cfg: Config, model: nn.Module) -> Trainer:
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), **cfg.optimizer.model_dump())
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, **cfg.scheduler.model_dump()
    )
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
    )
    return trainer
