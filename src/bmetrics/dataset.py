from dataclasses import dataclass
from functools import partial

from fairchem.core.datasets import LmdbDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, Subset
from torch_geometric.loader import DataLoader

from bmetrics.config import Config


@dataclass
class DataloaderSplits:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    train_val: DataLoader


def split_train_val_test(dataset, cfg: Config):
    train, temp = train_test_split(dataset, test_size=0.2, random_state=cfg.random_seed)
    val, test = train_test_split(temp, test_size=0.5, random_state=cfg.random_seed)
    return train, val, test


def get_dataloaders(cfg: Config):
    if cfg.paths.train and cfg.paths.val and cfg.paths.test:
        train = LmdbDataset({"src": str(cfg.paths.train)})
        val = LmdbDataset({"src": str(cfg.paths.val)})
        test = LmdbDataset({"src": str(cfg.paths.test)})
    else:
        dataset = LmdbDataset({"src": str(cfg.paths.data)})
        train, val, test = split_train_val_test(dataset, cfg)
    if cfg.fast_dev_run:
        indices = list(range(cfg.dataloader.batch_size))
        train = Subset(train, indices=indices)
        val = Subset(val, indices=indices)
        test = Subset(test, indices=indices)
    dataloader = partial(DataLoader, **cfg.dataloader.model_dump())
    train_val = ConcatDataset([train, val])
    return DataloaderSplits(
        train=dataloader(dataset=train, shuffle=True),
        val=dataloader(dataset=val, shuffle=False),
        test=dataloader(dataset=test, shuffle=False),
        train_val=dataloader(dataset=train_val, shuffle=True),
    )
