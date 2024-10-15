from pydantic import BaseModel


class Config(BaseModel):
    model_names: list[str]
    model_keys: list[str]
    lmdb_path: str
    output_path: str
