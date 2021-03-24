"""
Base agent class xonstains the base train, validate, test, inference, and utility functions
Other agents specific to a network overload the functions of this base agent class
"""
from modules.loggers.base_logger import get_logger
from easydict import EasyDict as edict
import os


class BaseAgent:
    """
    base functions which will be overloaded
    """

    def __init__(self, CONFIG, logger_name="agent"):
        """
        config is the edict configurations object
        """
        self.CONFIG = edict(CONFIG)
        logger_path = os.path.join(self.CONFIG.LOGGER.DIR,
                                   self.CONFIG.LOGGER.LOG_FMT.format(logger_name))
        self.logger = get_logger(logger_path)

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

    def finalize_exit(self):
        """
        operations for graceful exit
        """
        raise NotImplementedError
