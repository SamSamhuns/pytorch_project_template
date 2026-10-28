"""Image Classifier Data loader
"""
import os.path as osp

import webdataset as wds
from torchvision.transforms import Compose

from src.datasets import base_dataset
from src.utils.common import identity


def _get_webdataset_len(data_path) -> int:
    wdataset = (wds.WebDataset(data_path, shardshuffle=False)
                .decode("pil")
                .to_tuple("input.jpg", "output.cls"))
    length = 0
    for _ in wdataset:
        length += 1
    return length


def _get_webdataset(dset_path: str, tfms: Compose, dset_len: int, decode: str = "pil"):
    """Get a webdataset
    """
    return (wds.WebDataset(dset_path, shardshuffle=True)
            .shuffle(1000)
            .decode(decode)
            .to_tuple("input.jpg", "output.cls")
            .map_tuple(tfms, identity)
            .with_length(dset_len))


class ClassifierDataset:
    def __init__(self,
                 train_transform=None,
                 val_transform=None,
                 test_transform=None,
                 data_mode="imgs",
                 root='data',
                 train_path='train',
                 val_path='val',
                 test_path='test',
                 **kwargs):
        """train_transform: torchvision.transforms for train data
        val_transform: torchvision.transforms for validation data
        test_transform: torchvision.transforms for test data
        data_mode: Mode for getting data
        root: folder containing train & test data dirs
        train_path: train dir under root
        val_path: val dir under root
        test_path: test dir under root

        data_root
                |--train_path
                |--val_path
                |--test_path
        """
        self.val_set = None
        self.test_set = None
        if data_mode == "imgs":
            train_root = osp.join(root, train_path)
            self.train_set = base_dataset.ImageFolderDataset(
                train_root, transforms=train_transform)
            if val_path is not None:
                val_root = osp.join(root, val_path)
                self.val_set = base_dataset.ImageFolderDataset(
                    val_root, transforms=val_transform)
            if test_path is not None:
                test_root = osp.join(root, test_path)
                self.test_set = base_dataset.ImageFolderDataset(
                    test_root, transforms=test_transform)
            # class indices come from the sorted class dirs, so every split needs the same dirs
            for split in (self.val_set, self.test_set):
                if split is not None and split.classes != self.train_set.classes:
                    raise ValueError(f"Classes in {split.root} {split.classes} "
                                     f"differ from the train classes {self.train_set.classes}")
        elif data_mode == "webdataset":
            train_root = osp.join(root, train_path)
            train_len = _get_webdataset_len(train_root)
            self.train_set = _get_webdataset(train_root, train_transform, train_len)
            if val_path is not None:
                val_root = osp.join(root, val_path)
                val_len = _get_webdataset_len(val_root)
                self.val_set = _get_webdataset(val_root, val_transform, val_len)
            if test_path is not None:
                test_root = osp.join(root, test_path)
                test_len = _get_webdataset_len(test_root)
                self.test_set = _get_webdataset(test_root, test_transform, test_len)
        else:
            raise NotImplementedError(
                f"{data_mode} data_mode is not supported. Available modes are: imgs, webdataset")
