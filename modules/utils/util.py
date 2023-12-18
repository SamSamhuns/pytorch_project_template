"""
this util.py contains utility funcs meant to be used internally & externally
"""
import os
import json
import glob
import socket
import functools
from collections import OrderedDict
from collections.abc import MutableMapping
from easydict import EasyDict as edict


class BColors:
    """
    Border Color values for pretty printing in terminal
    Sample Use:
        print(f"{BColors.WARNING}Warning: Information.{BColors.ENDC}"
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MissingConfigError(Exception):
    """Raised when configurations are missing
    """

def is_port_in_use(port: int) -> bool:
    """
    Checks if a port is free for use
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        return stream.connect_ex(('localhost', int(port))) == 0


def identity(x):
    return x


def recursively_flatten_dict(dictionary, parent_key: str = '', sep: str = '.') -> MutableMapping:
    """
    Recursively flattens a potentially nested dict with nested keys seperated by sep
    """
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(recursively_flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def rgetattr(obj, attr, *args):
    """
    recursively get attrs. i.e. rgetattr(module, "sub1.sub2.sub3")
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def get_cfg_object(cfg_json_path: str):
    """
    generate configuration file from configs/base_configs.json JSON config file
    """
    cfg = _get_cfg_file(cfg_json_path)
    validate_base_config_dict(cfg)
    return cfg


def read_json(fname: str):
    with open(fname, 'r', encoding="utf-8") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: dict, fname: str) -> None:
    with open(fname, 'w', encoding="utf-8") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def find_latest_file_in_dir(dir_path: str, ext: str = "pth"):
    """returns latest file with the ext from the dir_path directory
    """
    dir_path = str(dir_path)
    dir_path_appended = dir_path + \
        (f"/*.{ext}" if dir_path[-1] != '/' else f"*.{ext}")
    list_of_files = glob.glob(dir_path_appended)
    if len(list_of_files) == 0:
        print(f"INFO: Directory '{dir_path}' is empty")
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def validate_base_config_dict(cfg: dict) -> None:
    """
    validates if required keys exist in cfg obj loaded from JSON config file
    """
    _required_top_keys = ['name', 'seed',
                          'cudnn_deterministic', 'cudnn_benchmark',
                          'gpu_device', 'arch', 'dataset',
                          'dataloader', 'optimizer', 'loss', 'metrics',
                          'lr_scheduler', 'trainer', 'logger']
    _arch = ["type", "args", "input_width", "input_height", "input_channel"]
    _dataset = ['type', 'args', 'num_classes', 'preprocess']
    _dataloader = ['type', 'args']
    _optimizer = ['type', 'args']
    _lr_scheduler = ['type', 'args']
    _trainer = ['resume_checkpoint', 'epochs', 'save_dir']

    _validate_keys(cfg, _required_top_keys)
    _validate_keys(cfg["arch"], _arch)
    _validate_keys(cfg["dataset"], _dataset)
    _validate_keys(cfg["dataloader"], _dataloader)
    _validate_keys(cfg["optimizer"], _optimizer)
    _validate_keys(cfg["lr_scheduler"], _lr_scheduler)
    _validate_keys(cfg["trainer"], _trainer)


# ################## Internal functions #######################


def _fix_path_for_globbing(directory: str):
    """ Add * at the end of paths for proper globbing
    """
    if directory[-1] == '/':         # data/
        directory += '*'
    elif directory[-1] != '*':       # data
        directory += '/*'
    else:                      # data/*
        directory = directory

    return directory


def _validate_keys(dictionary: dict, req_key_list: list) -> None:
    for key in req_key_list:
        if key not in dictionary.keys():
            raise MissingConfigError(
                f"key:'{key}' is missing from JSON config file")


def _get_cfg_file(cfg_json_path: str):
    """
    returns json cfg file as edict object
    """
    return edict(read_json(cfg_json_path))


if __name__ == "__main__":
    cfg = get_cfg_object("configs/classifier_cpu_config.json")
    print(cfg)
