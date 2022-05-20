# this util.py contains utility funcs meant to be used internally & externally
import os
import json
import glob
import functools
from collections import OrderedDict
from easydict import EasyDict as edict


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
    pass


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
    with open(fname, 'r') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: dict, fname: str) -> None:
    with open(fname, 'w') as handle:
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
    _required_top_keys = ['name', 'seed', 'use_cuda',
                          'cudnn_deterministic', 'cudnn_benchmark',
                          'gpu_device', 'arch', 'dataset',
                          'dataloader', 'optimizer', 'loss', 'metrics',
                          'lr_scheduler', 'trainer', 'logger']
    _arch = ["type", "args", "input_width", "input_height", "input_channel"]
    _dataset = ['type', 'args', 'num_classes', 'preprocess']
    _dataloader = ['type', 'args']
    _optimizer = ['type', 'args']
    _lr_scheduler = ['type', 'args']
    _trainer = ['resume', 'epochs', 'save_dir']

    _validate_keys(cfg, _required_top_keys)
    _validate_keys(cfg["arch"], _arch)
    _validate_keys(cfg["dataset"], _dataset)
    _validate_keys(cfg["dataloader"], _dataloader)
    _validate_keys(cfg["optimizer"], _optimizer)
    _validate_keys(cfg["lr_scheduler"], _lr_scheduler)
    _validate_keys(cfg["trainer"], _trainer)


# ################## Internal functions #######################


def _fix_path_for_globbing(dir: str):
    """ Add * at the end of paths for proper globbing
    """
    if dir[-1] == '/':         # data/
        dir += '*'
    elif dir[-1] != '*':       # data
        dir += '/*'
    else:                      # data/*
        dir = dir

    return dir


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
