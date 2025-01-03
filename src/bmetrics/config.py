from pydantic import BaseModel


class Config(BaseModel):
    random_seed: int
    data_root: str
    subset_size: int # 0 means no subset
    models_root: str
    device: str
    input_dim: int
    hidden_dim: int
    num_layers: int
    batch_size: int
    lr: float
    gamma: float
    max_epochs: int
    model_names: list
