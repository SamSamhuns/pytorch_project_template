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

from src.augmentations import init_transform
from src.config_parser import CustomDictConfig
from src.datasets import init_dataset
from src.utils.export_utils import ONNXDynamoExportStrategy, ONNXTSExportStrategy


class _BaseTrainer(ABC):
    """base functions which will be overloaded"""

    def __init__(self, config: CustomDictConfig, logger_name: str = "logger") -> None:
        self.config = config
        # the entry script sets up the console and file handlers once, with src.loggers.setup_logging
        self.logger = logging.getLogger(logger_name)

        # one device, e.g. cpu, cuda or cuda:1
        device = self.config["device"]
        if device.startswith("cuda") and not torch.cuda.is_available():
            self.logger.warning("CUDA is not available, so the program will run on CPU")
            device = "cpu"
        self.device = torch.device(device)
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

    def export(self):
        """Main export function"""
        raise NotImplementedError()


class BaseTrainer(_BaseTrainer):
    """Inherits from _BaseTrainer class with dataset, dataloader, and tboard logging initialized"""

    def __init__(self, config: CustomDictConfig, logger_name: str = "logger") -> None:
        super().__init__(config, logger_name)
        self.model = None  # placeholder model attrb. Must be init later
        # train_transform augments; eval_transform is shared by val, test and inference
        preprocess_cfg = self.config["dataset"]["preprocess"]
        self.preprocess = init_transform(preprocess_cfg["type"], **preprocess_cfg["args"])
        # do not load datasets for INFERENCE mode
        if self.config["mode"] in {"INFERENCE"}:
            return

        # ############### define dataset & dataloaders ################
        # a seeded generator makes the val split and the train shuffle reproducible
        generator = torch.Generator(device="cpu")
        if self.config["seed"] is not None:
            generator.manual_seed(self.config["seed"])
        # the val set comes from val_path, or validation_split carves it out of the train data
        self.data_set = init_dataset(
            self.config["dataset"]["type"],
            train_transform=self.preprocess.train_transform,
            eval_transform=self.preprocess.eval_transform,
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

    def export(self, mode: str = "ONNX_TS"):
        """Export pytorch model to onnx"""
        # dict with export path suffix & export strategy
        exporter_dict = {
            "ONNX_TS": ("_ts.onnx", ONNXTSExportStrategy),
            "ONNX_DYNAMO": ("_dynamo.onnx", ONNXDynamoExportStrategy),
        }
        if mode not in exporter_dict:
            raise NotImplementedError(f"{mode} export mode is not supported")
        if not self.model:
            raise NotImplementedError("Model not implemented/initialized")

        self.model.eval()
        self.logger.info("Initiated %s export mode", mode)

        in_w = self.config["model"]["info"]["input_width"]
        in_h = self.config["model"]["info"]["input_height"]
        in_c = self.config["model"]["info"]["input_channel"]
        sample_in = torch.randn((4, in_c, in_h, in_w)).to(self.device)

        model_name = "model_gpu" if self.device.type == "cuda" else "model_cpu"
        export_path = osp.join(self.config["models_dir"], model_name)
        export_path = export_path + exporter_dict[mode][0]
        exporter = exporter_dict[mode][1](self.logger)
        # export and test inference for equality with orig pytorch model
        exporter.export(self.model, export_path, sample_in)
        exporter.test_inference(self.model, export_path, sample_in)
        self.logger.info("%s export complete", mode)
