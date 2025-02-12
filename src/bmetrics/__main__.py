import json
import polars as pl
import toml
import torch

import wandb
from bmetrics.config import Config
from bmetrics.dataset import get_dataloaders
from bmetrics.experiment import make_experiment
from bmetrics.models import make_model
from bmetrics.train import make_trainer
from bmetrics.tune import tune_model


def main():
    cfg = Config(**toml.load("config.toml"))
    torch.manual_seed(cfg.random_seed)
    if cfg.log:
        wandb_cfg = cfg.model.model_dump() | cfg.optimizer.model_dump()
        wandb.init(project="Binding Model Metrics", config=wandb_cfg)
    loaders = get_dataloaders(cfg)
    experiment = make_experiment()
    results = []
    for params in experiment:
        if cfg.tune:
            tune_model(cfg, loaders)
        if cfg.train:
            try:
                with cfg.paths.hparams.open("r") as f:
                    hparams = json.load(f)
            except FileNotFoundError:
                hparams = None
            model = make_model(
                cfg,
                expert_names=params["experts"],
                moe=params["moe"],
                hparams=hparams,
            )
            trainer = make_trainer(cfg=cfg, model=model)
            trainer.train(loaders.train_val)
        if cfg.evaluate:
            model = torch.load(cfg.paths.checkpoint)
            trainer = make_trainer(cfg=cfg, model=model)
            test_loss = trainer.evaluate(loaders.test)
            result = params | {
                "train_mse": trainer.best_train_loss,
                "val_mse": trainer.best_val_loss,
                "test_mse": test_loss,
            }
            results.append(result)
            df = pl.DataFrame(results)
            df.write_parquet(cfg.paths.results)
    wandb.finish()


if __name__ == "__main__":
    main()
