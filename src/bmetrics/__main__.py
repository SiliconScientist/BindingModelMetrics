import toml
import torch
import torch.nn as nn
import torch.optim as optim
from fairchem.core.datasets import LmdbDataset
from torch.utils.data import Subset

import wandb
from bmetrics.config import Config
from bmetrics.dataset import split_train_val_test
from bmetrics.models import make_moe
from bmetrics.train import Trainer, evaluate


def main():
    config = Config(**toml.load("config.toml"))
    torch.manual_seed(config.random_seed)
    wandb_config = config.model.model_dump() | config.optimizer.model_dump()
    wandb.init(project="Binding Model Metrics", config=wandb_config)
    dataset = LmdbDataset({"src": str(config.paths.data)})
    if not config.subset_size == 0:
        dataset = Subset(dataset, indices=list(range(config.subset_size)))
    dataloaders = split_train_val_test(dataset, config)
    model = make_moe(config)
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
        config=config,
    )
    trainer.train()
    checkpoint = torch.load(config.paths.checkpoints)  # type: ignore
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test_loss = evaluate(
        model=model,
        dataloader=dataloaders.test,
        criterion=criterion,
        device=config.device,
    )
    print(f"Test Loss: {test_loss:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
