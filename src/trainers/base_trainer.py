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
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

from src.augmentations import init_transform
from src.config_parser import CustomDictConfig
from src.dataloaders import init_dataloader
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
        It holds the model, optimizer and scheduler state dicts, the epoch and best_val_acc.
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

        val_path = self.config["dataset"]["args"]["val_path"]
        test_path = self.config["dataset"]["args"]["test_path"]

        # ############# check exclusive config parameters #############
        val_split = self.config["dataloader"]["args"]["validation_split"]
        if val_path is not None and val_split > 0:
            msg = f"If val_path {val_path} is not None, val_split({val_split}) must be 0"
            self.logger.error(msg)
            raise ValueError(msg)

        # ############### define dataset & dataloaders ################
        self.data_set = init_dataset(
            self.config["dataset"]["type"],
            train_transform=self.preprocess.train_transform,
            val_transform=self.preprocess.eval_transform,
            test_transform=self.preprocess.eval_transform,
            **self.config["dataset"]["args"],
        )

        # log numerized remapped labels if present
        if self.config.verbose and hasattr(self.data_set, "labels2idx"):
            msg = f"Remapped labels dict = {self.data_set.labels2idx}"
            self.logger.info(msg)

        # define train, validate, and test data_loaders
        self.train_data_loader, self.val_data_loader, self.test_data_loader = None, None, None
        # in OSX systems ["dataloader"]["num_workers"] should be 0 which might increase train time
        dldr_type = self.config["dataloader"]["type"]
        data_ldr_args = dict(self.config["dataloader"]["args"].copy())

        # init torch.Generator for reproducibility if seed is present
        generator = torch.Generator(device="cpu")
        if self.config["seed"] is not None:
            generator.manual_seed(self.config["seed"])
        data_ldr_args["generator"] = generator

        self.train_data_loader = init_dataloader(dldr_type, dataset=self.data_set.train_set, **data_ldr_args)

        # if val_path is not None then dataloader.args.validation_split is assumed to be 0.0
        # if no val dir is provided, take val split from training data
        if val_path is None and data_ldr_args["validation_split"] > 0:
            self.val_data_loader = self.train_data_loader.get_validation_split()
        # if val dir is provided, use all data inside val dir for validation
        elif self.data_set.val_set is not None:
            self.val_data_loader = init_dataloader(dldr_type, dataset=self.data_set.val_set, **data_ldr_args)
        # if an invalid val_path is provided
        elif val_path is not None and not self.data_set.val_set:
            msg = f"val_path: {val_path} in config is invalid."
            self.logger.error(msg)
            raise ValueError(msg)

        if test_path is not None or self.data_set.test_set is not None:
            data_ldr_args["validation_split"] = 0.0
            if dldr_type != "WebDatasetDataLoader":
                data_ldr_args["shuffle"] = False
            self.test_data_loader = init_dataloader(dldr_type, dataset=self.data_set.test_set, **data_ldr_args)

        # log dataset data count, uniq labels and counts
        if self.config.verbose:
            ds = self.data_set.train_set
            train_set = ds.dataset if isinstance(ds, Subset) else ds
            test_set = self.data_set.test_set
            if dldr_type != "WebDatasetDataLoader":
                train_set = train_set.data
                test_set = self.data_set.test_set.data

            self.logger.info(
                "TRAIN: Loaded %d data points from %s", len(train_set), self.config["dataset"]["args"]["root"]
            )
            self.logger.info(
                "TEST: Loaded %d data points from %s", len(test_set), self.config["dataset"]["args"]["root"]
            )

            def _log_unique_label_counts(data_loader, label_type):
                """Extracts targets, flattens them, and logs unique counts."""
                targets = np.concatenate([t.numpy() for _, t in data_loader])
                uniq_elem, counts = np.unique(targets, return_counts=True)
                self.logger.info("%s dataset labels: %s, counts: %s", label_type, uniq_elem, counts)

            _log_unique_label_counts(self.train_data_loader, "Train")
            if self.val_data_loader:
                _log_unique_label_counts(self.val_data_loader, "Val")
            if self.test_data_loader:
                _log_unique_label_counts(self.test_data_loader, "Test")
        # #############################################################

        # ############ Tboard Summary Writer if enabled ##############
        if self.config["trainer"]["use_tensorboard"]:
            _agent_name = self.config["experiment_name"]
            _optim_name = self.config["optimizer"]["type"]
            _bsize = self.config["dataloader"]["args"]["batch_size"]
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
