import json
import torch
import bitsandbytes as bnb
from pathlib import Path
from torch import nn, optim
from torch_geometric.loader import DataLoader

import wandb
from bmetrics.config import Config
from bmetrics.dataset import DataloaderSplits
from bmetrics.models import make_model, set_hyperparameters


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: nn.MSELoss,
        scaler: torch.GradScaler,
        optimizer: bnb.optim.SGD,
        scheduler: optim.lr_scheduler.LRScheduler,
        cfg: Config,
    ):
        self.model = model
        self.criterion = criterion
        self.scaler = scaler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_train_loss = float("inf")
        self.best_val_loss = float("inf")
        self.cfg = cfg

    def train_step(self, data):
        self.optimizer.zero_grad(set_to_none=True)
        data = data.to(self.cfg.device)
        with torch.autocast(device_type=self.cfg.device, dtype=torch.float16):
            pred = self.model(data)
            loss = self.criterion(pred, data.energy)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
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
        total_loss = 0.0
        for data in dataloader:
            data = data.to(self.cfg.device)
            with torch.autocast(device_type=self.cfg.device, dtype=torch.float16):
                pred = self.model(data)
                loss = self.criterion(pred, data.energy)
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def save_checkpoint(self, path: Path) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )


def make_trainer(cfg: Config, model: nn.Module) -> Trainer:
    criterion = nn.MSELoss()
    scaler = torch.GradScaler(cfg.device)
    optimizer = bnb.optim.SGD(model.parameters(), **cfg.optimizer.model_dump())
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, **cfg.scheduler.model_dump()
    )
    trainer = Trainer(
        model=model,
        criterion=criterion,
        scaler=scaler,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
    )
    return trainer


def train_model(
    cfg: Config,
    expert_names: list[str],
    moe: bool,
    loaders: DataloaderSplits,
    use_best: bool = False,
):
    if use_best:
        with open(cfg.paths.hparams, "r") as file:
            hyperparameters = json.load(file)
        cfg = set_hyperparameters(cfg=cfg, **hyperparameters)
        model = make_model(cfg=cfg, expert_names=expert_names, moe=moe)
        trainer = make_trainer(cfg=cfg, model=model)
        trainer.train(loaders.train_val)
    else:
        model = make_model(cfg=cfg, expert_names=expert_names, moe=moe)
        trainer = make_trainer(cfg=cfg, model=model)
        trainer.train(loaders.train_val)
    torch.save(trainer.model.state_dict(), cfg.paths.checkpoint)
