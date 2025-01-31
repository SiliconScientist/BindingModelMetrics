from pathlib import Path

from pydantic import BaseModel


class Paths(BaseModel):
    data: Path
    train: Path
    val: Path
    test: Path
    experts: Path
    checkpoints: Path
    results: Path


class DataloaderConfig(BaseModel):
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool


class ModelConfig(BaseModel):
    hidden_dim: int
    input_dim: int
    num_layers: int


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


class Config(BaseModel):
    random_seed: int
    device: str
    fast_dev_run: bool
    log: bool
    paths: Paths
    dataloader: DataloaderConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    trainer: TrainerConfig
