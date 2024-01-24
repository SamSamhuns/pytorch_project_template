"""
Base agent class constains the base train, validate, test, inference, and utility functions
Other agents specific to a network overload the functions of this base agent class
"""
import glob
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn

from modules.config_parser import ConfigParser
import modules.datasets as module_datasets
import modules.dataloaders as module_dataloaders
import modules.augmentations as module_transforms
from modules.loggers.base_logger import get_logger
from modules.utils.statistics import print_cuda_statistics
from modules.utils.util import is_port_in_use, recursively_flatten_dict, rgetattr, find_latest_file_in_dir, BColors


class BaseAgent:
    """
    base functions which will be overloaded
    """

    def __init__(self, config: ConfigParser, logger_name: str = "logger") -> None:
        self.config = config
        # General logger
        self.logger = get_logger(logger_name=logger_name,
                                 logger_dir=self.config.log_dir,
                                 file_fmt=self.config["logger"]["file_fmt"],
                                 console_fmt=self.config["logger"]["console_fmt"],
                                 logger_level=self.config["logger"]["logger_level"],
                                 file_level=self.config["logger"]["file_level"],
                                 console_level=self.config["logger"]["console_level"])

        # ###################### set cuda vars ########################
        is_cuda = torch.cuda.is_available()
        gpu_device = self.config["gpu_device"]
        gpu_device = [gpu_device] if isinstance(gpu_device, int) else gpu_device
        if is_cuda and not gpu_device:
            msg = f"{BColors.WARNING}WARNING: CUDA available but not used{BColors.ENDC}"
            self.logger.info(msg)
        # set cuda devices if cuda available & gpu_device set or use cpu
        self.cuda = is_cuda and (gpu_device not in (None, []))
        self.manual_seed = self.config["seed"]
        torch.backends.cudnn.deterministic = self.config["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = self.config["cudnn_benchmark"]
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            if len(gpu_device) > 1 and torch.cuda.device_count() == 1:
                msg = f"Multi-gpu device ({gpu_device}) set, but only one GPU avai."
                self.logger.error(msg)
                raise ValueError(msg)
            # for dataparallel, only gpu_device[0] can be specified
            self.device = torch.device("cuda", gpu_device[0])
            self.logger.info("Program will run on GPU device %s", self.device)
            print_cuda_statistics()
        else:
            torch.manual_seed(self.manual_seed)
            self.device = torch.device("cpu")
            self.logger.info("Program will run on CPU")
        # #############################################################

        # check if num classes in cfg match num of avai class folders
        num_classes = self.config["dataset"]["num_classes"]
        data_root = self.config["dataset"]["args"]["data_root"]
        train_path = self.config["dataset"]["args"]["train_path"]
        val_path = self.config["dataset"]["args"]["val_path"]
        test_path = self.config["dataset"]["args"]["test_path"]
        train_count = len(glob.glob(osp.join(data_root, train_path, "*")))
        val_count = len(
            glob.glob(osp.join(data_root, val_path, "*"))) if val_path else 0
        test_count = len(
            glob.glob(osp.join(data_root, test_path, "*"))) if test_path else 0

        if config["mode"] in {"TRAIN", "TEST"}:
            warn_msg = f"{BColors.WARNING}WARNING: num_classes in cfg(%s) != avai classes in %s(%s){BColors.ENDC}"
            if train_count != num_classes:
                self.logger.info(warn_msg, num_classes,
                                 train_path, train_count)
            if val_path and val_count != num_classes:
                self.logger.info(warn_msg, num_classes, val_path, val_count)
            if test_path and test_count != num_classes:
                self.logger.info(warn_msg, num_classes, test_path, test_count)

        # check exclusive config parameters
        val_path = self.config["dataset"]["args"]["val_path"]
        val_split = self.config["dataloader"]["args"]["validation_split"]
        if (val_path is not None and val_split > 0):
            raise ValueError(
                f"If val_path {val_path} is not None, val_split({val_split}) must be 0")

        # do not load datasets for INFERENCE mode
        if config["mode"] in {"INFERENCE"}:
            return

        # ###################### define dataset #######################
        self.data_set = self.config.init_obj(
            "dataset", module_datasets,
            train_transform=rgetattr(
                module_transforms, self.config["dataset"]["preprocess"]["train_transform"]),
            val_transform=rgetattr(
                module_transforms, self.config["dataset"]["preprocess"]["val_transform"]),
            test_transform=rgetattr(
                module_transforms, self.config["dataset"]["preprocess"]["test_transform"]))
        # define train, validate, and test data_loaders
        self.train_data_loader, self.val_data_loader, self.test_data_loader = None, None, None
        # in OSX systems ["dataloader"]["num_workers"] should be 0 which might increase train time
        self.train_data_loader = self.config.init_obj("dataloader", module_dataloaders,
                                                      dataset=self.data_set.train_set)
        # if val_path is not None then dataloader.args.validation_split is assumed to be 0.0
        # if no val dir is provided, take val split from training data
        if val_path is None and self.config["dataloader"]["args"]["validation_split"] > 0:
            self.val_data_loader = self.train_data_loader.get_validation_split()
        # if val dir is provided, use all data inside val dir for validation
        elif val_path is not None or self.data_set.val_set is not None:
            self.val_data_loader = self.config.init_obj("dataloader", module_dataloaders,
                                                        dataset=self.data_set.val_set)
        if test_path is not None or self.data_set.test_set is not None:
            self.test_data_loader = self.config.init_ftn("dataloader", module_dataloaders,
                                                         dataset=self.data_set.test_set)
            self.test_data_loader = self.test_data_loader(validation_split=0.0)
        # #############################################################

        # ############ Tboard Summary Writer if enabled ###############
        if self.config["trainer"]["use_tensorboard"]:
            from torch.utils.tensorboard import SummaryWriter
            from tensorboard import program

            _agent_name = self.config["name"]
            _optim_name = self.config["optimizer"]["type"]
            _bsize = self.config["dataloader"]["args"]["batch_size"]
            _lr = self.config["optimizer"]["args"]["lr"]

            _suffix = f"{_agent_name}__{_optim_name}_BSIZE{_bsize}_LR{_lr}"
            tboard_log_dir = self.config["trainer"]["tensorboard_log_dir"]
            tboard_writer = SummaryWriter(log_dir=tboard_log_dir, filename_suffix=_suffix)
            self.tboard_writer = tboard_writer

            flat_cfg = recursively_flatten_dict(self.config._config)
            for cfg_key, cfg_val in flat_cfg.items():
                self.tboard_writer.add_text(cfg_key, str(cfg_val))

            tboard_port = self.config["trainer"]["tensorboard_port"]
            if tboard_port is not None:
                while is_port_in_use(tboard_port) and tboard_port < 65535:
                    tboard_port += 1
                    print(f"Port {tboard_port - 1} unavailable."
                          f"Switching to {tboard_port} for tboard logging")

                tboard = program.TensorBoard()
                tboard.configure(argv=[None, "--logdir", tboard_log_dir, "--port", str(tboard_port)])
                url = tboard.launch()
                print(f"Tensorboard logger started on {url}")
        # #############################################################

        # check if val_metrics present when save_best_only is True
        if self.config["trainer"]["save_best_only"] and not self.config["metrics"]["val"]:
            raise ValueError(
                "val metrics must be present to save best model")

    def load_checkpoint(self, model: nn.Module, file_path: str) -> nn.Module:
        """
        Load checkpoint for model from file_path
        args:
            model: model whose weights are loaded from file_path
            file_path: file_path to checkpoint file/folder with only weights,
                  if folder is used, latest checkpoint having extension 'ext' is loaded
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

        state_dict = torch.load(ckpt_file) if self.cuda else torch.load(
            ckpt_file, map_location=torch.device('cpu'))

        # rename keys for dataparallel mode
        if (self.config["gpu_device"] and len(self.config["gpu_device"]) > 1 and torch.cuda.device_count() > 1):
            _state_dict = OrderedDict()
            for k, val in state_dict.items():
                k = 'module.' + \
                    k if 'module' not in k else k.replace(
                        'features.module.', 'module.features.')
                _state_dict[k] = val
            state_dict = _state_dict

        model.load_state_dict(state_dict)
        self.logger.info("Loaded checkpoint %s", ckpt_file)
        return model

    def save_checkpoint(self, model: nn.Module, ckpt_save_name: str = "checkpoint.pth") -> None:
        """
        Checkpoint saver
        args:
            ckpt_save_name: checkpoint file name which is saved inside self.config.save_dir
        """
        # create checkpoint directory if it doesnt exist
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        save_path = osp.join(str(self.config.save_dir), ckpt_save_name)
        torch.save(model.state_dict(), save_path)

    def train(self):
        """
        main train function
        """
        raise NotImplementedError

    def train_one_epoch(self):
        """
        train function to run for one epoch
        """
        raise NotImplementedError

    def validate(self):
        """
        main validate function
        """
        raise NotImplementedError

    def test(self):
        """
        main test function
        """
        raise NotImplementedError

    def inference(self):
        """
        inference function
        """
        raise NotImplementedError

    def export_as_onnx(self, dummy_input: torch.Tensor, onnx_save_path: str):
        """
        ONNX format export function
        """
        raise NotImplementedError

    def finalize_exit(self):
        """
        operations for graceful exit
        """
        raise NotImplementedError
