"""
Base agent class constains the base train, validate, test, inference, and utility functions
Other agents specific to a network overload the functions of this base agent class
"""
from modules.loggers.base_logger import get_logger
from easydict import EasyDict as edict


class BaseAgent:
    """
    base functions which will be overloaded
    """

    def __init__(self, CONFIG, logger_fname="train", logger_name="logger") -> None:
        """
        config is the edict configurations object
        """
        self.CONFIG = edict(CONFIG)
        # General logger
        self.logger = get_logger(logger_fname=self.CONFIG.LOGGER.LOG_FMT.format(logger_fname),
                                 logger_dir=self.CONFIG.LOGGER.DIR,
                                 logger_name=logger_name,
                                 file_fmt=self.CONFIG.LOGGER.FILE_FMT,
                                 console_fmt=self.CONFIG.LOGGER.CONSOLE_FMT,
                                 logger_level=self.CONFIG.LOGGER.LOGGER_LEVEL,
                                 file_level=self.CONFIG.LOGGER.FILE_LEVEL,
                                 console_level=self.CONFIG.LOGGER.CONSOLE_LEVEL)
        # Tboard Summary Writer if enabled
        if self.CONFIG.TRAINER.USE_TENSORBOARD:
            from torch.utils.tensorboard import SummaryWriter
            from datetime import datetime

            _agent_name = self.CONFIG.NAME
            _optim_name = self.CONFIG.OPTIMIZER.TYPE.__name__
            _bsize = self.CONFIG.DATALOADER.BATCH_SIZE
            _lr = self.CONFIG.OPTIMIZER.LR

            _suffix = f"{_agent_name}__{_optim_name}_BSIZE{_bsize}_LR{_lr}"
            _tboard_log_root_dir = self.CONFIG.TRAINER.TENSORBOARD_EXPERIMENT_DIR
            _tboard_log_sub_dir = _suffix + datetime.now().strftime("_%Y_%m_%d__%H_%M_%S")
            _tboard_log_dir = _tboard_log_root_dir + '/' + _tboard_log_sub_dir
            tboard_writer = SummaryWriter(log_dir=_tboard_log_dir,
                                          filename_suffix=_suffix)
            self.tboard_writer = tboard_writer

        # check exclusive config parameters
        val_dir, val_split = self.CONFIG.DATASET.VAL_DIR, self.CONFIG.DATALOADER.VALIDATION_SPLIT
        if (val_dir is not None and val_split > 0):
            raise RuntimeError(f"If VAL_DIR {val_dir} is not None, {val_split} must be 0")

    def load_checkpoint(self, file_name):
        """
        load latest checkpoint file
        :param file_name: checkpoint file_name (pth, pt, pkl file)
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=False):
        """
        save checkpoint
        :param file_name: checkpoint file_name (pth, pt, pkl file)
        """
        raise NotImplementedError

    def run(self):
        """
        function to call train, validate or test mode
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
