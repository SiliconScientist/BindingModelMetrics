import toml
import torch
from fairchem.core.datasets import LmdbDataset
from torch.utils.data import Subset

import wandb
from bmetrics.config import Config
from bmetrics.dataset import split_train_val_test
from bmetrics.train import make_trainer


def main():
    config = Config(**toml.load("config.toml"))
    torch.manual_seed(config.random_seed)
    wandb_config = config.model.model_dump() | config.optimizer.model_dump()
    wandb.init(project="Binding Model Metrics", config=wandb_config)
    dataset = LmdbDataset({"src": str(config.paths.data)})
    if not config.subset_size == 0:
        dataset = Subset(dataset, indices=list(range(config.subset_size)))
    dataloaders = split_train_val_test(dataset, config)
    trainer = make_trainer(config=config, dataloaders=dataloaders)
    trainer.train()
    test_loss = trainer.test()
    print(f"Test Loss: {test_loss:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
