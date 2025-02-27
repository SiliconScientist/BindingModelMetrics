import polars as pl
import toml
import torch

import wandb
from bmetrics.config import Config
from bmetrics.dataset import get_dataloaders
from bmetrics.experiment import make_experiment
from bmetrics.models import make_model
from bmetrics.train import make_trainer


def main():
    config = Config(**toml.load("config.toml"))
    torch.manual_seed(config.random_seed)
    if config.log:
        wandb_config = config.model.model_dump() | config.optimizer.model_dump()
        wandb.init(project="Binding Model Metrics", config=wandb_config)
    dataloaders = get_dataloaders(config)
    experiment = make_experiment()
    results = []
    for params in experiment:
        model = make_model(config, expert_names=params["experts"], moe=params["moe"])
        trainer = make_trainer(config, dataloaders, model)
        if params["finetune"]:
            trainer.train()
        test_loss = trainer.test()
        result = params | {
            "train_mse": trainer.best_train_loss,
            "val_mse": trainer.best_val_loss,
            "test_mse": test_loss,
        }
        results.append(result)
        df = pl.DataFrame(results)
        df.write_parquet(config.paths.results)
    wandb.finish()


if __name__ == "__main__":
    main()
