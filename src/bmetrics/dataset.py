from dataclasses import dataclass
from functools import partial
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader


@dataclass
class DataloaderSplits:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def split_train_val_test(dataset, config):
    train, temp = train_test_split(
        dataset, test_size=0.3, random_state=config.random_seed
    )
    val, test = train_test_split(temp, test_size=0.5, random_state=config.random_seed)
    dataloader = partial(DataLoader, **config.dataloader.model_dump())
    return DataloaderSplits(
        train=dataloader(dataset=train, shuffle=True),
        val=dataloader(dataset=val, shuffle=False),
        test=dataloader(dataset=test, shuffle=False),
    )
