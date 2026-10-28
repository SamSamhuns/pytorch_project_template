"""Logger object

Logging levels
Debug = 10
Info = 20
Warning = 30
Error = 40
Critical = 50
"""

import logging
import os


def get_logger(
    logger_name: str,
    logger_dir: str = "logs",
    file_fmt: str = "%(asctime)s %(levelname)-8s: %(message)s",
    console_fmt: str = "%(message)s",
    logger_level: int = logging.DEBUG,
    file_level: int = logging.DEBUG,
    console_level: int = logging.DEBUG,
    propagate: bool = False,
) -> logging.Logger:
    """Logger settings should be configured and imported from configs dir
    Note: adding multiple handlers to a logger causes the same logs being printed multiple times
    """
    os.makedirs(logger_dir, exist_ok=True)
    filename = os.path.join(logger_dir, logger_name + ".txt")

    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)
    logger.propagate = propagate

    file_fmt = logging.Formatter(file_fmt)
    log_file = logging.FileHandler(filename)
    log_file.setLevel(file_level)
    log_file.setFormatter(file_fmt)
    logger.addHandler(log_file)

    console_fmt = logging.Formatter(console_fmt)
    log_console = logging.StreamHandler()
    log_console.setLevel(console_level)
    log_console.setFormatter(console_fmt)
    logger.addHandler(log_console)

    return logger
