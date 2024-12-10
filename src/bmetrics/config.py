from pydantic import BaseModel


class Config(BaseModel):
    random_seed: int
    data_root: str
    weights_root: str
    device: str
    input_dim: int
    hidden_channels: int
    batch_size: int
    learning_rate: float
    num_epochs: int
    model_names: list
