"""Base Trainer
Implements logging torch device setup, checkpoint loading,
tensorboard logging, dataset and dataloader inits
"""

import logging
import os.path as osp
import textwrap
from abc import ABC, abstractmethod

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, IterableDataset, Subset
from torch.utils.tensorboard import SummaryWriter

from src.augmentations import build_transforms
from src.config_parser import CustomDictConfig
from src.datasets import DATASETS
from src.utils.common import get_device


class _BaseTrainer(ABC):
    """base functions which will be overloaded"""

    def __init__(self, config: CustomDictConfig, logger_name: str = "logger") -> None:
        self.config = config
        # the entry script sets up the console and file handlers once, with src.loggers.setup_logging
        self.logger = logging.getLogger(logger_name)

        self.device = get_device(self.config["device"])
        self.logger.info("Program will run on %s", self.device)

    def load_checkpoint(self, file_path: str) -> dict:
        """Load a checkpoint saved by save_checkpoint, e.g. models/best.pth, onto the CPU.
        It holds the model, optimizer and scheduler state dicts, the epoch, best_val_acc and the class names.
        """
        if not osp.isfile(file_path):
            msg = f"'{file_path}' is not a checkpoint file. Pass one, e.g. .../models/best.pth"
            self.logger.error(msg)
            raise ValueError(msg)

        checkpoint = torch.load(file_path, map_location="cpu", weights_only=True)
        self.logger.info("Loaded checkpoint %s", file_path)
        return checkpoint

    @abstractmethod
    def train(self):
        """Main train function"""
        raise NotImplementedError()

    @abstractmethod
    def test(self):
        """Main test function"""
        raise NotImplementedError()

    def validate(self):
        """Main validate function"""
        raise NotImplementedError()


class BaseTrainer(_BaseTrainer):
    """Inherits from _BaseTrainer class with dataset, dataloader, and tboard logging initialized"""

    def __init__(self, config: CustomDictConfig, logger_name: str = "logger") -> None:
        super().__init__(config, logger_name)
        self.model = None  # placeholder model attrb. Must be init later
        # train_transform augments; eval_transform is shared by val, test and inference
        train_transform, eval_transform, _ = build_transforms(self.config["dataset"]["preprocess"])

        # ############### define dataset & dataloaders ################
        # a seeded generator makes the val split and the train shuffle reproducible
        generator = torch.Generator(device="cpu")
        if self.config["seed"] is not None:
            generator.manual_seed(self.config["seed"])
        # the val set comes from val_path, or validation_split carves it out of the train data
        self.data_set = DATASETS[self.config["dataset"]["type"]](
            train_transform=train_transform,
            eval_transform=eval_transform,
            generator=generator,
            **self.config["dataset"]["args"],
        )

        num_workers = self.config["dataloader"]["num_workers"]
        loader_args = {
            "batch_size": self.config["dataloader"]["batch_size"],
            "num_workers": num_workers,
            "persistent_workers": num_workers > 0,  # start the workers once, not every epoch
            "pin_memory": self.device.type == "cuda",  # page-locked memory speeds up copies to the GPU
        }
        # a WebDataset shuffles itself; the DataLoader shuffles the other (map-style) datasets
        shuffle = not isinstance(self.data_set.train_set, IterableDataset)
        # drop_last, as BatchNorm raises on a last train batch of one sample
        self.train_data_loader = DataLoader(
            self.data_set.train_set, shuffle=shuffle, drop_last=True, generator=generator, **loader_args
        )
        self.val_data_loader, self.test_data_loader = None, None
        if self.data_set.val_set is not None:
            self.val_data_loader = DataLoader(self.data_set.val_set, **loader_args)
        if self.data_set.test_set is not None:
            self.test_data_loader = DataLoader(self.data_set.test_set, **loader_args)

        # log the data count and label counts of each split, from the targets rather than a pass over the images
        if self.config.verbose:
            root = self.config["dataset"]["args"]["root"]
            splits = {"Train": self.data_set.train_set, "Val": self.data_set.val_set, "Test": self.data_set.test_set}
            for split, dataset in splits.items():
                if dataset is None:
                    continue
                self.logger.info("%s: Loaded %d data points from %s", split, len(dataset), root)
                if isinstance(dataset, IterableDataset):
                    continue  # a WebDataset reads each label with its image
                if isinstance(dataset, Subset):  # carved out of the train data by validation_split
                    targets = np.asarray(dataset.dataset.targets)[dataset.indices]
                else:
                    targets = np.asarray(dataset.targets)
                labels, counts = np.unique(targets, return_counts=True)
                self.logger.info("%s dataset labels: %s, counts: %s", split, labels, counts)
        # #############################################################

        # ############ Tboard Summary Writer if enabled ##############
        if self.config["trainer"]["use_tensorboard"]:
            _agent_name = self.config["experiment_name"]
            _optim_name = self.config["optimizer"]["type"]
            _bsize = self.config["dataloader"]["batch_size"]
            _lr = self.config["optimizer"]["args"]["lr"]

            _suffix = f"{_agent_name}__{_optim_name}_BSIZE{_bsize}_LR{_lr}"

            tb_logdir = self.config.tboard_log_dir
            tboard_writer = SummaryWriter(log_dir=tb_logdir, filename_suffix=_suffix)
            self.tboard_writer = tboard_writer
            # the whole config as YAML, indented so TensorBoard's markdown shows it as a code block
            self.tboard_writer.add_text("config", textwrap.indent(OmegaConf.to_yaml(self.config), "    "))
        # #############################################################
