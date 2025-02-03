import numpy as np
import optuna
import polars as pl
from optuna.pruners import HyperbandPruner
from optuna.samplers import QMCSampler, TPESampler

from bmetrics.config import Config
from bmetrics.models import make_model
from bmetrics.train import make_trainer


def make_params(trial: optuna.Trial, config: Config):
    config = config.model_copy(deep=True)
    hparams = config.hyperparameters
    config.model.hidden_dim = trial.suggest_int(**hparams.model.hidden_dim)
    config.model.num_layers = trial.suggest_int(**hparams.model.num_layers)
    config.model.dropout = trial.suggest_float(**hparams.model.dropout)
    config.optimizer.lr = trial.suggest_float(**hparams.optimizer.lr)
    config.optimizer.weight_decay = trial.suggest_float(
        **hparams.optimizer.weight_decay
    )
    return config


class Objective:
    def __init__(self, config: Config, dataloaders):
        self.config = config
        self.dataloaders = dataloaders
        self.best_value = float("inf")

    def __call__(self, trial: optuna.Trial):
        config = make_params(trial, config=self.config)
        params = {
            "experts": ["dimenetpp", "schnet", "painn"],
            "finetune": True,
            "moe": True,
        }
        model = make_model(
            config=config, expert_names=params["experts"], moe=params["moe"]
        )
        trainer = make_trainer(config=config, model=model, dataloaders=self.dataloaders)
        trainer.train()
        val_loss = trainer.best_val_loss
        val_loss = float("inf") if np.isnan(val_loss) else val_loss
        trainer.test()
        if (val_loss < self.best_value) and (~np.isnan(val_loss)):
            self.best_value = val_loss
            path = config.paths.checkpoints / "best.ckpt"
            trainer.save_checkpoint(path)
        return val_loss


def tune_model(config: Config, data_module):
    sampler = QMCSampler(seed=config.random_seed)
    pruner = HyperbandPruner(
        min_resource=config.tuner.min_resource,
        max_resource=config.trainer.max_epochs,
    )
    storage = optuna.storages.RDBStorage(
        url="sqlite:///:memory:",
        engine_kwargs={"pool_size": 10, "connect_args": {"timeout": 10}},
    )
    study = optuna.create_study(
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        study_name="ABCD",
    )
    objective = Objective(config=config, data_module=data_module)
    half_trials = config.tuner.n_trials // 2
    study.optimize(func=objective, n_trials=half_trials)
    study.sampler = TPESampler(multivariate=True, seed=config.random_seed)
    study.optimize(func=objective, n_trials=half_trials)
    df = pl.DataFrame(study.trials_dataframe())
    df.write_parquet(config.filepaths.data.results.study)
    return study
