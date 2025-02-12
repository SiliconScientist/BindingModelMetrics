import json

import optuna
import polars as pl
import torch
import torch.nn as nn
from optuna.samplers import QMCSampler, TPESampler

from bmetrics.config import Config
from bmetrics.models import MixtureOfExperts, make_model, set_hyperparameters
from bmetrics.train import make_trainer


def make_params(trial: optuna.Trial, cfg: Config):
    cfg = cfg.model_copy(deep=True)
    hparams = cfg.hyperparameters
    cfg.model.hidden_dim = trial.suggest_int(**hparams.model.hidden_dim)
    cfg.model.num_layers = trial.suggest_int(**hparams.model.num_layers)
    cfg.model.dropout = trial.suggest_float(**hparams.model.dropout)
    cfg.optimizer.lr = trial.suggest_float(**hparams.optimizer.lr)
    cfg.optimizer.weight_decay = trial.suggest_float(**hparams.optimizer.weight_decay)
    cfg.trainer.max_epochs = trial.suggest_int(**hparams.trainer.max_epochs)
    return cfg


class Objective:
    def __init__(self, cfg: Config, loaders):
        self.cfg = cfg
        self.loaders = loaders
        self.best_value = float("inf")

    def __call__(self, trial: optuna.Trial):
        cfg = make_params(trial, cfg=self.cfg)
        params = {
            "experts": ["dimenetpp", "schnet", "painn"],
            "moe": True,
        }
        model = make_model(cfg=cfg, expert_names=params["experts"], moe=params["moe"])
        trainer = make_trainer(cfg=cfg, model=model)
        trainer.train(self.loaders.train, self.loaders.val)
        if trainer.best_val_loss < self.best_value:
            self.best_value = trainer.best_val_loss
            torch.save(trainer.model.state_dict(), cfg.paths.checkpoint)
        return trainer.best_val_loss


def tune_model(cfg: Config, loaders):
    sampler = QMCSampler(seed=cfg.random_seed)
    storage = optuna.storages.RDBStorage(
        url="sqlite:///:memory:",
        engine_kwargs={"pool_size": 10, "connect_args": {"timeout": 10}},
    )
    study = optuna.create_study(
        storage=storage,
        sampler=sampler,
        direction="minimize",
        study_name="Mamun",
    )
    objective = Objective(cfg=cfg, loaders=loaders)
    half_trials = cfg.tuner.n_trials // 2
    study.optimize(func=objective, n_trials=half_trials)
    study.sampler = TPESampler(multivariate=True, seed=cfg.random_seed)
    study.optimize(func=objective, n_trials=half_trials)
    df = pl.DataFrame(study.trials_dataframe()).sort("value", descending=True)
    df.write_parquet(cfg.paths.study)
    with open(cfg.paths.hparams, "w") as f:
        json.dump(study.best_params, f)
    return study


def load_best_checkpoint(cfg: Config, expert_names: list[str], moe: bool) -> nn.Module:
    model_weights = torch.load(cfg.paths.checkpoint, weights_only=True)
    with open(cfg.paths.hparams, "r") as file:
        hyperparams = json.load(file)
    cfg = set_hyperparameters(cfg=cfg, **hyperparams)
    model: nn.Module = make_model(cfg=cfg, expert_names=expert_names, moe=moe)
    model.load_state_dict(model_weights)
    model.to(cfg.device)
    return model
