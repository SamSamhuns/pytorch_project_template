"""
Base Trainer
Implements logging torch device setup, save & load checkpoint,
tensorboard logging, dataset and dataloader inits
"""
import os
import os.path as osp
from abc import ABC, abstractmethod
from collections import OrderedDict
import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset
from torchvision.transforms.transforms import Normalize
from tensorboard import program
from omegaconf import OmegaConf

from src.loggers import get_logger
from src.config_parser import CustomDictConfig
from src.datasets import init_dataset
from src.dataloaders import init_dataloader
from src.augmentations import init_transform
from src.utils.export_utils import (
    ONNXDynamoExportStrategy, ONNXTSExportStrategy,
    TSScriptExportStrategy, TSTraceExportStrategy,
    QuantizedModelWrapper)
from src.utils.common import (
    recursively_flatten_config, is_port_in_use,
    find_latest_file_in_dir, BColors)


class _BaseTrainer(ABC):
    """
    base functions which will be overloaded
    """

    def __init__(self, config: CustomDictConfig, logger_name: str = "logger") -> None:
        self.config = config
        # General logger
        self.logger = get_logger(
            logger_name=logger_name, logger_dir=self.config["logs_dir"])

        # ###################### set cuda vars ########################
        is_cuda = torch.cuda.is_available() and self.config["device"] == "cuda"
        gpu_dev = self.config["gpu_device"]
        gpu_dev = [gpu_dev] if isinstance(gpu_dev, int) else gpu_dev
        if is_cuda and not gpu_dev:
            msg = f"{BColors.WARN}WARNING: CUDA available but not used{BColors.ENDC}"
            self.logger.info(msg)
        # set cuda devices if cuda available & gpu_dev set or use cpu
        self.cuda = is_cuda and (gpu_dev not in (None, []))
        torch.backends.cudnn.deterministic = self.config["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = self.config["cudnn_benchmark"]
        if self.cuda:
            torch.cuda.manual_seed(self.config["seed"])
            if len(gpu_dev) > 1 and torch.cuda.device_count() == 1:
                msg = f"Multi-gpu device ({gpu_dev}) set, but only one GPU avai."
                self.logger.error(msg)
                raise ValueError(msg)
            # for dataparallel, only gpu_dev[0] can be specified
            self.device = torch.device("cuda", gpu_dev[0])
            self.logger.info("Program will run on GPU device %s", self.device)
        else:
            torch.manual_seed(self.config["seed"])
            self.device = torch.device("cpu")
            self.logger.info("Program will run on CPU")
        # #############################################################

    def load_checkpoint(self, model: nn.Module, file_path: str) -> nn.Module:
        """
        Load checkpoint for model from file_path
        args:
            model: model whose weights are loaded from file_path
            file_path: file_path to checkpoint file/folder with only weights,
                  if folder is used, latest checkpoint having extension "ext" is loaded
        """
        ckpt_file = None
        if osp.isfile(file_path):
            ckpt_file = file_path
        elif osp.isdir(file_path):
            ckpt_file = find_latest_file_in_dir(file_path, ext="pth")
        if ckpt_file is None:
            msg = f"'{file_path}' is not a torch weight file or a dir containing one."
            self.logger.error(msg)
            raise ValueError(msg)

        wts_only = True
        state_dict = torch.load(ckpt_file, weights_only=wts_only) if self.cuda else torch.load(
            ckpt_file, map_location=torch.device("cpu"), weights_only=wts_only)

        # rename keys for dataparallel mode
        use_cuda = self.config["device"] == "cuda"
        gpu_dev = self.config["gpu_device"]
        if (use_cuda and gpu_dev and len(gpu_dev) > 1 and torch.cuda.device_count() > 1):
            _state_dict = OrderedDict()
            for k, val in state_dict.items():
                k = "module." + \
                    k if "module" not in k else k.replace(
                        "features.module.", "module.features.")
                _state_dict[k] = val
            state_dict = _state_dict

        model.load_state_dict(state_dict)
        self.logger.info("Loaded checkpoint %s", ckpt_file)
        return model

    def save_checkpoint(self, model: nn.Module, ckpt_save_name: str = "checkpoint.pth") -> None:
        """
        Checkpoint saver
        args:
            ckpt_save_name: checkpoint file name which is saved inside self.config["models_dir"]
        """
        # create checkpoint directory if it doesnt exist
        os.makedirs(self.config["models_dir"], exist_ok=True)
        save_path = osp.join(str(self.config["models_dir"]), ckpt_save_name)
        torch.save(model.state_dict(), save_path)

    @abstractmethod
    def train(self):
        """
        main train function
        """
        raise NotImplementedError()

    @abstractmethod
    def test(self):
        """
        main test function
        """
        raise NotImplementedError()

    def validate(self):
        """
        main validate function
        """
        raise NotImplementedError()

    def export(self):
        """
        main export function
        """
        raise NotImplementedError()


class BaseTrainer(_BaseTrainer):
    """
    Inherits from _BaseTrainer class with dataset, dataloader, and tboard logging initialized
    """

    def __init__(self, config: CustomDictConfig, logger_name: str = "logger") -> None:
        super().__init__(config, logger_name)
        self.model = None  # placeholder model attrb. Must be init later
        # do not load datasets for INFERENCE mode
        if config["mode"] in {"INFERENCE"}:
            return

        val_path = self.config["dataset"]["args"]["val_path"]
        test_path = self.config["dataset"]["args"]["test_path"]

        # ############# check exclusive config parameters #############
        val_split = self.config["dataloader"]["args"]["validation_split"]
        if (val_path is not None and val_split > 0):
            msg = f"If val_path {val_path} is not None, val_split({val_split}) must be 0"
            self.logger.error(msg)
            raise ValueError(msg)
        # check if val_metrics present when save_best_only is True
        if self.config["trainer"]["save_best_only"] and not self.config["metrics"]["val"]:
            msg = "val metrics must be present to save best model"
            self.logger.error(msg)
            raise ValueError(msg)

        # ############### define dataset & dataloaders ################
        self.data_set = init_dataset(
            self.config["dataset"]["type"],
            train_transform=init_transform(
                self.config["dataset"]["preprocess"]["train_transform"]).train,
            val_transform=init_transform(
                self.config["dataset"]["preprocess"]["val_transform"]).val,
            test_transform=init_transform(
                self.config["dataset"]["preprocess"]["test_transform"]).test,
            **self.config["dataset"]["args"]
        )

        # update train+test transforms in config and save to YAML file
        train_tfs = self.data_set.train_transform
        test_tfs = self.data_set.test_transform
        self.config["dataset"]["args"]["train_tfs"] = [str(tfs) for tfs in train_tfs.transforms]
        self.config["dataset"]["args"]["test_tfs"] = [str(tfs) for tfs in test_tfs.transforms]
        normalize = [tfs for tfs in train_tfs.transforms if isinstance(tfs, Normalize)]
        self.config["dataset"]["args"]["mean"] = normalize[0].mean
        self.config["dataset"]["args"]["std"] = normalize[0].std
        _config = dict(self.config)
        # save updated config to YAML file
        OmegaConf.save(_config, osp.join(config["save_dir"], "config.yaml"))

        # log numerized remapped labels if present
        if config.verbose and hasattr(self.data_set, "labels2idx"):
            msg = f"Remapped labels dict = {self.data_set.labels2idx}"
            self.logger.info(msg)

        # define train, validate, and test data_loaders
        self.train_data_loader, self.val_data_loader, self.test_data_loader = None, None, None
        # in OSX systems ["dataloader"]["num_workers"] should be 0 which might increase train time
        dldr_type = self.config["dataloader"]["type"]
        data_ldr_args = self.config["dataloader"]["args"].copy()
        self.train_data_loader = init_dataloader(
            dldr_type, dataset=self.data_set.train_set, **data_ldr_args)

        # if val_path is not None then dataloader.args.validation_split is assumed to be 0.0
        # if no val dir is provided, take val split from training data
        if val_path is None and data_ldr_args["validation_split"] > 0:
            self.val_data_loader = self.train_data_loader.get_validation_split()
        # if val dir is provided, use all data inside val dir for validation
        elif self.data_set.val_set is not None:
            self.val_data_loader = init_dataloader(
                dldr_type, dataset=self.data_set.val_set, **data_ldr_args)
        # if an invalid val_path is provided
        elif val_path is not None and not self.data_set.val_set:
            msg = f"val_path: {val_path} in config is invalid."
            self.logger.error(msg)
            raise ValueError(msg)

        if test_path is not None or self.data_set.test_set is not None:
            data_ldr_args["validation_split"] = 0.0
            if dldr_type != "WebDatasetDataLoader":
                data_ldr_args["shuffle"] = False
            self.test_data_loader = init_dataloader(
                dldr_type, dataset=self.data_set.test_set, **data_ldr_args)

        # log dataset data count, uniq labels and counts
        if config.verbose:
            ds = self.data_set.train_set
            train_set = ds.dataset if isinstance(ds, Subset) else ds
            test_set = self.data_set.test_set
            if dldr_type != "WebDatasetDataLoader":
                train_set = train_set.data
                test_set = self.data_set.test_set.data

            self.logger.info(
                "TRAIN: Loaded %d data points from %s",
                len(train_set), self.config["dataset"]["args"]["root"])
            self.logger.info(
                "TEST: Loaded %d data points from %s",
                len(test_set), self.config["dataset"]["args"]["root"])

            def _log_unique_label_counts(data_loader, label_type):
                """Extracts targets, flattens them, and logs unique counts."""
                targets = np.concatenate([t.numpy() for _, t in data_loader])
                uniq_elem, counts = np.unique(targets, return_counts=True)
                self.logger.info(
                    "%s dataset labels: %s, counts: %s", label_type, uniq_elem, counts)

            _log_unique_label_counts(self.train_data_loader, "Train")
            if self.val_data_loader:
                _log_unique_label_counts(self.val_data_loader, "Val")
            if self.test_data_loader:
                _log_unique_label_counts(self.test_data_loader, "Test")
        # #############################################################

        # ############ Tboard Summary Writer if enabled ##############
        if self.config["trainer"]["use_tensorboard"]:
            _agent_name = self.config["name"]
            _optim_name = self.config["optimizer"]["type"]
            _bsize = self.config["dataloader"]["args"]["batch_size"]
            _lr = self.config["optimizer"]["args"]["lr"]

            _suffix = f"{_agent_name}__{_optim_name}_BSIZE{_bsize}_LR{_lr}"

            tb_logdir = config.tboard_log_dir
            tboard_writer = SummaryWriter(
                log_dir=tb_logdir, filename_suffix=_suffix)
            self.tboard_writer = tboard_writer

            flat_cfg = recursively_flatten_config(self.config)
            for cfg_key, cfg_val in flat_cfg.items():
                self.tboard_writer.add_text(cfg_key, str(cfg_val))

            tboard_port = self.config["trainer"]["tensorboard_port"]
            if tboard_port is not None:
                while is_port_in_use(tboard_port) and tboard_port < 65535:
                    tboard_port += 1
                    print(f"Port {tboard_port - 1} unavailable."
                          "Switching to {tboard_port} for tboard logging")

                tboard = program.TensorBoard()
                tboard.configure(
                    argv=[None, "--logdir", tb_logdir, "--port", str(tboard_port)])
                url = tboard.launch()
                self.logger.info("Tensorboard logger started on %s", url)
        # #############################################################

    def export(self, mode: str = "ONNX_TS", quantize_mode_backend: str = None):
        """
        Export pytorch model to onnx/torchscript with optional quantization
        Currently supported quantization mode includes ("fbgemm", "x86", "qnnpack", "onednn")
        """
        # dict with export path suffix & export strategy
        exporter_dict = {
            "ONNX_TS": ("_ts.onnx", ONNXTSExportStrategy),
            "ONNX_DYNAMO": ("_dynamo.onnx", ONNXDynamoExportStrategy),
            "TS_TRACE": ("_traced.ptc", TSTraceExportStrategy),
            "TS_SCRIPT": ("_scripted.ptc", TSScriptExportStrategy)
        }
        if mode not in exporter_dict:
            raise NotImplementedError(f"{mode} export mode is not supported")
        if not self.model:
            raise NotImplementedError("Model not implemented/initialized")
        if quantize_mode_backend:
            self.logger.info("Init model quantization with %s backend", quantize_mode_backend)
            self.model = QuantizedModelWrapper(self.model)
            self.model = self.quantize_model(quantize_mode_backend)

        self.model.eval()
        self.logger.info("Initiated %s export mode", mode)

        in_w = self.config["model"]["info"]["input_width"]
        in_h = self.config["model"]["info"]["input_height"]
        in_c = self.config["model"]["info"]["input_channel"]
        sample_in = torch.randn((4, in_c, in_h, in_w)).to(self.device)

        model_name = "model_gpu" if self.device.type == "cuda" else "model_cpu"
        export_path = osp.join(self.config["models_dir"], model_name)
        export_path += f"_quant_{quantize_mode_backend}" if quantize_mode_backend else ""
        export_path = export_path + exporter_dict[mode][0]
        exporter = exporter_dict[mode][1](self.logger)
        # export and test inference for equality with orig pytorch model
        exporter.export(self.model, export_path, sample_in)
        exporter.test_inference(self.model, export_path, sample_in)
        self.logger.info("%s export complete", mode)

    def quantize_model(self, backend: str = "qnnpack") -> nn.Module:
        """
        Quantizes the model.
        Args:
            backend (str): The backend for quantization ("fbgemm", "x86", "qnnpack", "onednn").
        Note: All tensors must be in cpu
        """
        if self.device != torch.device("cpu"):
            self.logger.warning("Quantization is only tested on cpu. Running on GPU may cause errors.")
        self.model.eval()
        torch.backends.quantized.engine = backend
        self.model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
        torch.ao.quantization.prepare(self.model, inplace=True)

        # Calibrate the model using the test loader
        self.logger.info("Calibrating quantized model with test data...")
        with torch.no_grad():
            for data, _ in tqdm.tqdm(self.test_data_loader):
                data = data.to(self.device, non_blocking=True)
                self.model(data)

        quantized_model = torch.ao.quantization.convert(self.model, inplace=False)
        self.logger.info("Quantization complete with %s backend", backend)
        return quantized_model
