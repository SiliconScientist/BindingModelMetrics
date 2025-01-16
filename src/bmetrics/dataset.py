from dataclasses import dataclass
from functools import partial
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from fairchem.core.datasets import LmdbDataset


from bmetrics.config import Config


@dataclass
class DataloaderSplits:
    train: DataLoader
    val: DataLoader
    cal: DataLoader
    test: DataLoader


def split_train_val_test(dataset, config: Config):
    train, temp = train_test_split(
        dataset, test_size=0.1, random_state=config.random_seed
    )
    temp, test = train_test_split(temp, test_size=0.5, random_state=config.random_seed)
    val, cal = train_test_split(temp, test_size=0.5, random_state=config.random_seed)
    return train, val, cal, test


def get_dataloaders(config: Config):
    if (
        config.paths.train
        and config.paths.val
        and config.paths.cal
        and config.paths.test
    ):
        train = LmdbDataset({"src": str(config.paths.train)})
        val = LmdbDataset({"src": str(config.paths.val)})
        cal = LmdbDataset({"src": str(config.paths.cal)})
        test = LmdbDataset({"src": str(config.paths.test)})
    else:
        dataset = LmdbDataset({"src": str(config.paths.data)})
        train, val, cal, test = split_train_val_test(dataset, config)
    if config.fast_dev_run:
        indices = list(range(config.dataloader.batch_size))
        train = Subset(train, indices=indices)
        val = Subset(val, indices=indices)
        cal = Subset(cal, indices=indices)
        test = Subset(test, indices=indices)
    dataloader = partial(DataLoader, **config.dataloader.model_dump())
    return DataloaderSplits(
        train=dataloader(dataset=train, shuffle=True),
        val=dataloader(dataset=val, shuffle=False),
        cal=dataloader(dataset=cal, shuffle=False),
        test=dataloader(dataset=test, shuffle=False),
    )
