import json
import polars as pl
from sympy import false
import toml
import torch

import wandb
from bmetrics.config import Config
from bmetrics.dataset import get_dataloaders
from bmetrics.experiment import make_experiment
from bmetrics.train import make_trainer, train_model
from bmetrics.tune import tune_model, load_best_checkpoint


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
            train_model(
                cfg=cfg,
                expert_names=params["experts"],
                moe=params["moe"],
                loaders=loaders,
                use_best=True,
            )
        if cfg.evaluate:
            model = load_best_checkpoint(
                cfg=cfg, expert_names=params["experts"], moe=params["moe"]
            )
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
