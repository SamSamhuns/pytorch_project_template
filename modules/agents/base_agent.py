"""
Base agent class constains the base train, validate, test, inference, and utility functions
Other agents specific to a network overload the functions of this base agent class
"""
from modules.loggers.base_logger import get_logger
from modules.config_parser import ConfigParser


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

            _agent_name = self.config["name"]
            _optim_name = self.config["optimizer"]["type"]
            _bsize = self.config["dataloader"]["args"]["batch_size"]
            _lr = self.config["optimizer"]["args"]["lr"]

            _suffix = f"{_agent_name}__{_optim_name}_BSIZE{_bsize}_LR{_lr}"
            _tboard_log_root_dir = self.config["trainer"]["tensorboard_log_dir"]
            _tboard_log_dir = _tboard_log_root_dir
            tboard_writer = SummaryWriter(log_dir=_tboard_log_dir,
                                          filename_suffix=_suffix)
            self.tboard_writer = tboard_writer

        # check exclusive config parameters
        val_path = self.config["dataset"]["args"]["val_path"]
        val_split = self.config["dataloader"]["args"]["validation_split"]
        if (val_path is not None and val_split > 0):
            raise RuntimeError(
                f"If val_path {val_path} is not None, val_split({val_split}) must be 0")

    def load_checkpoint(self, file_path: str):
        """
        load latest checkpoint file
        args:
            file_path: checkpoint file_path (pth, pt, pkl file)
        """
        raise NotImplementedError

    def save_checkpoint(self, file_path: str = "checkpoint.pth.tar", is_best: bool = False):
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
