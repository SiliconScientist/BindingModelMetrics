import numpy as np
import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader

import wandb
from bmetrics.config import Config
from bmetrics.criterion import ReducedQuantileLoss
from bmetrics.dataset import DataloaderSplits


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: ReducedQuantileLoss,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cal_loader: DataLoader,
        test_loader: DataLoader,
        config: Config,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cal_loader = cal_loader
        self.test_loader = test_loader
        self.best_train_loss = float("inf")
        self.best_val_loss = float("inf")
        self.qhat = 0.0
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
                self.best_train_loss = train_loss
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
        weights = torch.load(self.config.paths.checkpoints, weights_only=True)
        self.model.load_state_dict(weights["model_state_dict"])

    @torch.no_grad()
    def evaluate(self, dataloader) -> float:
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

    @torch.no_grad()
    def calibrate(self, alpha: float = 0.1) -> None:
        self.model.eval()
        cal_scores = []
        for data in self.cal_loader:
            data = data.to(self.config.device)
            pred = self.model(data)
            score = torch.max(data.energy - pred[:, 0], pred[:, 1] - data.energy)
            cal_scores.append(score)
        cal_scores = torch.cat(cal_scores, dim=0).numpy()
        n = cal_scores.shape[0]
        qhat = np.quantile(
            cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher"
        )
        self.qhat = qhat

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()  # Set the model to evaluation mode
        predictions = []
        y_labels = []
        for data in loader:
            data = data.to(self.config.device)
            pred = self.model(data)
            y_label = data.energy
            predictions.append(pred)
            y_labels.append(y_label)
        predictions = torch.cat(predictions)
        y_labels = torch.cat(y_labels)
        return predictions, y_labels

    def conformalize(self):
        self.calibrate()
        predictions, y_labels = self.predict(self.test_loader)
        prediction_set = torch.stack(
            [predictions[:, 0] - self.qhat, predictions[:, 1] + self.qhat],
            dim=1,
        )
        return prediction_set, y_labels


def make_trainer(
    config: Config, dataloaders: DataloaderSplits, model: nn.Module
) -> Trainer:
    criterion = ReducedQuantileLoss(alpha=0.1)
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
        cal_loader=dataloaders.cal,
        test_loader=dataloaders.test,
        config=config,
    )
    return trainer
