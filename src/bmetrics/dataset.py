from dataclasses import dataclass
from functools import partial
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset


from bmetrics.config import Config


@dataclass
class DataloaderSplits:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def split_train_val_test(dataset, config: Config):
    train, temp = train_test_split(
        dataset, test_size=0.2, random_state=config.random_seed
    )
    val, test = train_test_split(temp, test_size=0.5, random_state=config.random_seed)
    if config.fast_dev_run:
        indices = list(range(config.dataloader.batch_size))
        train = Subset(train, indices=indices)
        val = Subset(val, indices=indices)
        test = Subset(test, indices=indices)
    dataloader = partial(DataLoader, **config.dataloader.model_dump())
    return DataloaderSplits(
        train=dataloader(dataset=train, shuffle=True),
        val=dataloader(dataset=val, shuffle=False),
        test=dataloader(dataset=test, shuffle=False),
    )
