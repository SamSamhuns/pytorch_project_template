"""
Base agent class constains the base train, validate, test, inference, and utility functions
Other agents specific to a network overload the functions of this base agent class
"""
import torch

from modules.config_parser import ConfigParser
from modules.loggers.base_logger import get_logger
from modules.utils.statistics import print_cuda_statistics
from modules.utils.util import is_port_in_use, recursively_flatten_dict, BColors


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
        # Tboard Summary Writer if enabled
        if self.config["trainer"]["use_tensorboard"]:
            from torch.utils.tensorboard import SummaryWriter
            from tensorboard import program

            _agent_name = self.config["name"]
            _optim_name = self.config["optimizer"]["type"]
            _bsize = self.config["dataloader"]["args"]["batch_size"]
            _lr = self.config["optimizer"]["args"]["lr"]

            _suffix = f"{_agent_name}__{_optim_name}_BSIZE{_bsize}_LR{_lr}"
            _tboard_log_dir = self.config["trainer"]["tensorboard_log_dir"]
            tboard_writer = SummaryWriter(log_dir=_tboard_log_dir,
                                          filename_suffix=_suffix)
            self.tboard_writer = tboard_writer

            flat_cfg = recursively_flatten_dict(self.config._config)
            for cfg_key, cfg_val in flat_cfg.items():
                self.tboard_writer.add_text(cfg_key, str(cfg_val))

            _tboard_port = self.config["trainer"]["tensorboard_port"]
            if _tboard_port is not None:
                while is_port_in_use(_tboard_port) and _tboard_port < 65535:
                    _tboard_port += 1
                    print(f"Port {_tboard_port - 1} unavailable." \
                          f"Switching to {_tboard_port} for tboard logging")

                tboard = program.TensorBoard()
                tboard.configure(
                    argv=[None, "--logdir", _tboard_log_dir, "--port", str(_tboard_port)])
                url = tboard.launch()
                print(f"Tensorboard logger started on {url}")
        # check if val_metrics present when save_best_only is True
        if self.config["trainer"]["save_best_only"] and not self.config["metrics"]["val"]:
            raise ValueError(
                "val metrics must be present to save best model")

        # set cuda flag
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
            if len(gpu_device) > 1 and torch.cuda.device_count() == 1:
                msg = f"Multi-gpu device ({gpu_device}) set, but only one GPU avai."
                self.logger.error(msg)
                raise ValueError(msg)
            torch.cuda.manual_seed(self.manual_seed)
            # for dataparallel, only gpu_device[0] can be specified
            self.device = torch.device("cuda", gpu_device[0])
            self.logger.info("Program will run on GPU device %s", self.device)
            print_cuda_statistics()
        else:
            torch.manual_seed(self.manual_seed)
            self.device = torch.device("cpu")
            self.logger.info("Program will run on CPU")

        # check exclusive config parameters
        val_path = self.config["dataset"]["args"]["val_path"]
        val_split = self.config["dataloader"]["args"]["validation_split"]
        if (val_path is not None and val_split > 0):
            raise ValueError(
                f"If val_path {val_path} is not None, val_split({val_split}) must be 0")

    def load_checkpoint(self, file_path: str):
        """
        load latest checkpoint file
        args:
            file_path: checkpoint file_path (pth, pt, pkl file)
        """
        raise NotImplementedError

    def save_checkpoint(self, file_path: str = "checkpoint.pth.tar"):
        """
        save checkpoint
        args:
            file_path: checkpoint file_path (pth, pt, pkl file)
        """
        raise NotImplementedError

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

    def export_as_onnx(self):
        """
        ONNX format export function
        """
        raise NotImplementedError

    def finalize_exit(self):
        """
        operations for graceful exit
        """
        raise NotImplementedError
