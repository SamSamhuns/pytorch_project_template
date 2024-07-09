"""
Logger object

Logging levels
Debug = 10
Info = 20
Warning = 30
Error = 40
Critical = 50
"""
import os
import pprint
import logging


class LogFilter(logging.Filter):
    """
    Filters logger logs based on exclusion keyword
    """

    def __init__(self, exc_keyword):
        super().__init__()
        self.exc_keyword = exc_keyword

    def filter(self, record):
        # Only allow messages NOT containing the exc_keyword
        # to pass through to the console
        return self.exc_keyword not in record.getMessage()


def get_logger(logger_name: str,
               logger_dir: str = 'logs',
               file_fmt: str = '%(asctime)s %(levelname)-8s: %(message)s',
               console_fmt: str = '%(message)s',
               logger_level: int = logging.DEBUG,
               file_level: int = logging.DEBUG,
               console_level: int = logging.DEBUG,
               console_exc_keyword: str = "",
               propagate: bool = False) -> logging.Logger:
    """
    Logger settings should be configured and imported from configs dir
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
    if console_exc_keyword:
        log_console.addFilter(LogFilter(console_exc_keyword))
    logger.addHandler(log_console)

    return logger


if __name__ == "__main__":
    opt = {'model': 'test', 'stage': 'train', 'epochs': 300}

    clogger = get_logger("test_logger", console_exc_keyword="NO CONSOLE")
    clogger.info('\n\nOptions:')
    clogger.info(pprint.pformat(opt))
    clogger.info("THIS SHOULD NOT APPEAR IN TERMINAL! NO CONSOLE")
