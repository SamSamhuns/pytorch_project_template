# this util.py contains utility funcs meant to be used internally & externally
import os
import json
import glob
import torch
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


def get_cfg_object(cfg_json_path):
    """
    generate configuration file from configs/base_configs.json JSON config file
    """
    cfg = _get_cfg_file(cfg_json_path)
    _validate_base_configurations(cfg)

    os.makedirs(cfg.TRAINER.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.LOGGER.DIR, exist_ok=True)
    return cfg


def read_json(fname):
    with open(fname, 'r') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with open(fname, 'w') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def prepare_device(n_gpu_use):
    """setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def find_latest_file_in_dir(dir_path, ext=".pth"):
    """returns latest file with the ext from the dir_path directory
    """
    dir_path_appended = dir_path + \
        (f"/*.{ext}" if dir_path[-1] != '/' else f"*.{ext}")
    list_of_files = glob.glob(dir_path_appended)
    if len(list_of_files) == 0:
        print(f"Directory '{dir_path}' is empty")
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

################### Internal functions #######################


def _fix_path_for_globbing(dir):
    """ Add * at the end of paths for proper globbing
    """
    if dir[-1] == '/':         # data/
        dir += '*'
    elif dir[-1] != '*':       # data
        dir += '/*'
    else:                      # data/*
        dir = dir

    return dir


def _validate_base_configurations(cfg):
    """validates if required keys exist in cfg obj
        loaded from JSON config file
    """
    _required_top_keys = ['NAME', 'SEED', 'USE_CUDA', 'N_GPU',
                          'GPU_DEVICE', 'ARCH', 'DATASET',
                          'DATALOADER', 'OPTIMIZER', 'LOSS', 'METRICS',
                          'LR_SCHEDULER', 'TRAINER', 'LOGGER']
    _arch = ['TYPE', 'PRETRAINED', 'INPUT_WIDTH', 'INPUT_HEIGHT']
    _dataset = ['RAW_DATA_ROOT_DIR', 'PROC_DATA_ROOT_DIR',
                'TRAIN_DIR', 'TEST_DIR', 'NUM_CLASSES']
    _dataloader = ['TYPE', 'BATCH_SIZE', 'SHUFFLE']
    _optimizer = ['TYPE', 'LR']
    _lr_scheduler = ['TYPE']
    _trainer = ['RESUME', 'EPOCHS', 'CHECKPOINT_DIR']
    _logger = ['DIR']

    _validate_keys(cfg, _required_top_keys)
    _validate_keys(cfg.ARCH, _arch)
    _validate_keys(cfg.DATASET, _dataset)
    _validate_keys(cfg.DATALOADER, _dataloader)
    _validate_keys(cfg.OPTIMIZER, _optimizer)
    _validate_keys(cfg.LR_SCHEDULER, _lr_scheduler)
    _validate_keys(cfg.TRAINER, _trainer)
    _validate_keys(cfg.LOGGER, _logger)


def _validate_keys(dictionary, req_key_list):
    for key in req_key_list:
        if key not in dictionary.keys():
            raise MissingConfigError(
                f"{key} is missing from JSON config file")


def _get_cfg_file(cfg_json_path):
    """
    returns json cfg file as edict object
    """
    return edict(read_json(cfg_json_path))


if __name__ == "__main__":
    cfg = get_cfg_object("configs/base_configs.json")
    print(cfg)
