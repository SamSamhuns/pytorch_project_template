import os
import pprint
import logging


def get_logger(logger_name,
               logger_dir='logs',
               file_fmt='%(asctime)s %(levelname)-8s: %(message)s',
               console_fmt='%(message)s',
               logger_level=logging.DEBUG,
               file_level=logging.DEBUG,
               console_level=logging.DEBUG):
    """
    Logger settings should be configured and imported from configs dir
    """
    os.makedirs(logger_dir, exist_ok=True)
    filename = os.path.join(logger_dir, logger_name)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)
    logger.propagate = False

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


if __name__ == "__main__":
    opt = {'model': 'test', 'stage': 'train', 'epochs': 300}

    logger = get_logger("test_logger")
    logger.info('\n\nOptions:')
    logger.info(pprint.pformat(opt))
