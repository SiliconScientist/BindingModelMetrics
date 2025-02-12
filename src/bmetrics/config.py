from pathlib import Path

from pydantic import BaseModel


class Paths(BaseModel):
    data: Path
    train: Path
    val: Path
    test: Path
    experts: Path
    checkpoint: Path
    results: Path
    study: Path
    hparams: Path


class DataloaderConfig(BaseModel):
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool


class ModelConfig(BaseModel):
    hidden_dim: int
    num_layers: int
    dropout: float


class ModelHParams(BaseModel):
    hidden_dim: dict
    num_layers: dict
    dropout: dict


class OptimizerHParams(BaseModel):
    lr: dict
    weight_decay: dict


class TrainerHParams(BaseModel):
    max_epochs: dict


class Hyperparameters(BaseModel):
    model: ModelHParams
    optimizer: OptimizerHParams
    trainer: TrainerHParams


class OptimizerConfig(BaseModel):
    lr: float
    weight_decay: float
    momentum: float
    nesterov: bool


class SchedulerConfig(BaseModel):
    T_max: int
    eta_min: float


class TrainerConfig(BaseModel):
    max_epochs: int


class TunerConfig(BaseModel):
    n_trials: int


class Config(BaseModel):
    random_seed: int
    device: str
    fast_dev_run: bool
    tune: bool
    train: bool
    evaluate: bool
    log: bool
    paths: Paths
    dataloader: DataloaderConfig
    model: ModelConfig
    hyperparameters: Hyperparameters
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    trainer: TrainerConfig
    tuner: TunerConfig
