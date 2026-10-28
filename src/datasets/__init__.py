import logging

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterableDataset, Subset

from .classifier_dataset import ClassifierDataset
from .mnist_dataset import MnistDataset

logger = logging.getLogger(__name__)

# the dataset.type values a config can name
DATASETS = {"ClassifierDataset": ClassifierDataset, "MnistDataset": MnistDataset}


def build_dataloaders(
    dataset, cfg: DictConfig, pin_memory: bool = False, generator: torch.Generator | None = None
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    """DataLoaders for the train, val and test splits of dataset, or None for a split it doesn't have.
    cfg: the dataloader section of a config, with batch_size and num_workers
    pin_memory: page-locked memory speeds up copies to a GPU
    generator: seeds the shuffle of the train loader
    """
    loader_args = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "persistent_workers": cfg.num_workers > 0,  # start the workers once, not every epoch
        "pin_memory": pin_memory,
    }
    # a WebDataset shuffles itself; the DataLoader shuffles the other (map-style) datasets.
    # drop_last, as BatchNorm raises on a last train batch of one sample
    shuffle = not isinstance(dataset.train_set, IterableDataset)
    train_loader = DataLoader(dataset.train_set, shuffle=shuffle, drop_last=True, generator=generator, **loader_args)
    val_loader, test_loader = (
        None if split is None else DataLoader(split, **loader_args) for split in (dataset.val_set, dataset.test_set)
    )
    return train_loader, val_loader, test_loader


def log_label_counts(dataset) -> None:
    """Log the size and the label counts of each split, from the targets rather than a pass over the images"""
    for name, split in (("Train", dataset.train_set), ("Val", dataset.val_set), ("Test", dataset.test_set)):
        if split is None:
            continue
        logger.info("%s: %d data points", name, len(split))
        if isinstance(split, IterableDataset):
            continue  # a WebDataset reads each label with its image
        if isinstance(split, Subset):  # carved out of the train data by validation_split
            targets = np.asarray(split.dataset.targets)[split.indices]
        else:
            targets = np.asarray(split.targets)
        labels, counts = np.unique(targets, return_counts=True)
        logger.info("%s labels: %s, counts: %s", name, labels, counts)
