from pathlib import Path

from pydantic import BaseModel


class Paths(BaseModel):
    data: Path
    models: Path
    checkpoints: Path


class DataloaderConfig(BaseModel):
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool


class ModelConfig(BaseModel):
    names: list[str]
    hidden_dim: int
    input_dim: int
    num_layers: int


class OptimizerConfig(BaseModel):
    lr: float
    momentum: float
    nesterov: bool


class SchedulerConfig(BaseModel):
    T_max: int
    eta_min: float


class TrainerConfig(BaseModel):
    max_epochs: int


class Config(BaseModel):
    random_seed: int
    subset_size: int  # 0 means no subset
    device: str
    dataloader: DataloaderConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    trainer: TrainerConfig
    paths: Paths
