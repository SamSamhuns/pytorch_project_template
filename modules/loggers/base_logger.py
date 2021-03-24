import pprint
import logging


def get_logger(filename,
               logger_name='logger',
               file_fmt='%(asctime)s %(levelname)-8s: %(message)s',
               console_fmt='%(message)s',
               logger_level=logging.DEBUG,
               file_level=logging.DEBUG,
               console_level=logging.DEBUG):

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

    logger = get_logger("logs/sample_log.txt")
    logger.info('\n\nOptions:')
    logger.info(pprint.pformat(opt))
