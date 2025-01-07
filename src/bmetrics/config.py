from pathlib import Path

from pydantic import BaseModel


class Paths(BaseModel):
    data: Path
    models: Path
    checkpoints: Path


class Config(BaseModel):
    random_seed: int
    subset_size: int  # 0 means no subset
    device: str
    input_dim: int
    hidden_dim: int
    num_layers: int
    batch_size: int
    lr: float
    momentum: float
    nesterov: bool
    gamma: float
    max_epochs: int
    model_names: list
    paths: Paths
