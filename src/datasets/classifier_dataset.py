"""Image Classifier Data loader"""

import glob
import json
import os.path as osp

import webdataset as wds
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import Compose

from src.utils.common import identity

# scripts/convert_dataset_to_tar.py stores each image under its file extension, e.g. jpg or png
IMAGE_KEYS = ";".join(extension[1:] for extension in IMG_EXTENSIONS)


class _WebDataset(wds.WebDataset):
    """A WebDataset with len(). wds' with_length() creates a class at runtime that can't be pickled, and DataLoader
    workers pickle the dataset when they start with spawn (macOS) or forkserver (Linux from Python 3.14)"""

    def __len__(self):
        return self.num_samples


def _get_webdataset(split_dir: str, tfms: Compose, train: bool = False) -> _WebDataset:
    """The shards and info.json that scripts/convert_dataset_to_tar.py wrote to split_dir"""
    with open(osp.join(split_dir, "info.json"), encoding="utf-8") as info_file:
        info = json.load(info_file)
    shards = sorted(glob.glob(osp.join(split_dir, "*.tar")))
    # each DataLoader worker reads its own shards; empty_check=False lets a worker without one read nothing.
    # Training shuffles the order of the shards and of the samples, within a buffer of 1000
    dataset = _WebDataset(shards, shardshuffle=len(shards) if train else False, empty_check=False)
    if train:
        dataset = dataset.shuffle(1000)
    dataset = dataset.decode("pil").to_tuple(IMAGE_KEYS, "cls").map_tuple(tfms, identity)
    dataset.num_samples = info["num_samples"]  # so len() needs no pass over the images
    dataset.classes = info["classes"]
    return dataset


class ClassifierDataset:
    def __init__(
        self,
        train_transform=None,
        eval_transform=None,
        data_mode="imgs",
        root="data",
        train_path="train",
        val_path="val",
        test_path="test",
        validation_split=0.0,
        generator=None,
    ):
        """train_transform: torchvision.transforms for train data
        eval_transform: torchvision.transforms for validation and test data
        data_mode: imgs for one folder of images per class, webdataset for scripts/convert_dataset_to_tar.py shards
        root: folder containing train & test data dirs
        train_path: train dir under root
        val_path: val dir under root
        test_path: test dir under root
        validation_split: without a val_path, the fraction of the train data to validate on
        generator: torch.Generator that seeds the validation split

        data_root
                |--train_path
                |--val_path
                |--test_path
        """
        if val_path is not None and validation_split > 0:
            raise ValueError(f"Set either val_path ({val_path}) or validation_split ({validation_split}), not both")
        if data_mode == "webdataset" and validation_split > 0:
            raise ValueError("A WebDataset can't split off a val set; convert a separate val split instead")
        self.val_set = None
        self.test_set = None
        if data_mode == "imgs":
            self.train_set = ImageFolder(osp.join(root, train_path), transform=train_transform)
            if val_path is not None:
                self.val_set = ImageFolder(osp.join(root, val_path), transform=eval_transform)
            if test_path is not None:
                self.test_set = ImageFolder(osp.join(root, test_path), transform=eval_transform)
        elif data_mode == "webdataset":
            self.train_set = _get_webdataset(osp.join(root, train_path), train_transform, train=True)
            if val_path is not None:
                self.val_set = _get_webdataset(osp.join(root, val_path), eval_transform)
            if test_path is not None:
                self.test_set = _get_webdataset(osp.join(root, test_path), eval_transform)
        else:
            raise NotImplementedError(f"{data_mode} data_mode is not supported. Available modes are: imgs, webdataset")

        # class indices come from the sorted class dirs, so every split needs the same dirs
        for split_path, split in ((val_path, self.val_set), (test_path, self.test_set)):
            if split is not None and split.classes != self.train_set.classes:
                raise ValueError(
                    f"Classes in {osp.join(root, split_path)} {split.classes} "
                    f"differ from the train classes {self.train_set.classes}"
                )
        self.classes = self.train_set.classes  # the class names, in the order of the model's outputs

        if validation_split > 0:
            self.train_set, self.val_set = random_split(
                self.train_set, [1 - validation_split, validation_split], generator=generator
            )
            # validate on the same images loaded with the eval transform, so they aren't augmented
            self.val_set.dataset = ImageFolder(osp.join(root, train_path), transform=eval_transform)
