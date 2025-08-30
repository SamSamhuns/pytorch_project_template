"""Contains utility functions meant to be used internally & externally
"""
import glob
import io
import os
import socket
import subprocess
from collections import OrderedDict
from collections.abc import Callable, MutableMapping
from contextlib import redirect_stdout
from typing import Any, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms.transforms import Compose


class BColors:
    """Border Color values for pretty printing in terminal
    Sample Use:
        print(f"{BColors.WARN}Warning: Information.{BColors.ENDC}"
    """

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARN = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


############################ conversion utils ############################


def try_bool(val: Any) -> bool | None:
    """Check if x is boolabe (eq. to true/false) & converts to bool else raise ValueError
    """
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    raise ValueError(
        f"{val} is not a boolean string. Must be 'true' or 'false' converted in lowecase.")


def try_null(val: Any) -> Optional[None]:
    """Check if x is nullable (eq. to "null") & converts to None else raise ValueError
    """
    if val.lower() in {"null", "none"}:
        return None
    raise ValueError(
        f"{val} is not a nullable string. Must be 'null' or 'none'  converted in lowecase.")


def can_be_conv_to_float(var: Any) -> bool:
    """Checks if a var can be converted to a float & return bool
    """
    try:
        float(var)
        return True
    except ValueError:
        return False


####################################################################


def get_git_revision_hash() -> str:
    """Get the git hash of the current commit. Returns None if run from a non-git init repo"""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except subprocess.CalledProcessError as excep:
        print(excep, "Couldn't get git hash of the current repo. Returning None")
    return None


def capture_output(func: Callable, *args, **kwargs):
    """Capture the stdout of a function call"""
    f = io.StringIO()
    with redirect_stdout(f):
        func(*args, **kwargs)
    return f.getvalue()


def identity(x):
    return x


def is_port_in_use(port: int) -> bool:
    """Checks if a port is free for use
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        return stream.connect_ex(('localhost', int(port))) == 0


def round_to_nearest_divisor(n: int, divisor: int) -> int:
    """Rounds an integer to the nearest multiple of a given divisor.

    Parameters
    ----------
        n (int): The integer to be rounded.
        divisor (int): The divisor to which 'n' will be rounded to the nearest multiple.

    Returns
    -------
        int: The nearest multiple of 'divisor' to 'n'.

    """
    mod = n % divisor
    if mod > (divisor//2):
        return n + (divisor - mod)
    return n - mod


def stable_sort(arr: list[Any]) -> list:
    """Sorts lists with numbers which are of type strings in the correct order.
    Numeric elements will appear before non-numeric.
    i.e. ['1', '10', '2'] is sorted to ['1', '2', '10'],
         [7.0, '1', '2', '10', 5, 'xyz', 'a'] is sorted to ['1', '2', 5, 7.0, '10', 'a', 'xyz'],
         ['2', 'a', 'c', 'b', '1'] is sorted to ['1', '2', 'a', 'b', 'c']
    """
    # Segregate elements based on whether they can be converted to float
    numeric_elements = [x for x in arr if can_be_conv_to_float(x)]
    non_numeric_elements = [x for x in arr if not can_be_conv_to_float(x)]

    return sorted(numeric_elements, key=float) + sorted(non_numeric_elements)


def sigmoid(x):
    """Numerically stable sigmoid"""
    return np.exp(-np.logaddexp(0, -x))


############################ dict utils ############################


def inherit_missing_dict_params(
        parent: dict, child: dict, ignore_keys: set[str], inplace: bool = True) -> None | dict:
    """Inherit missing params from parent to child dict, ignoring ignore_keys

    Parameters
    ----------
        parent (dict): The dictionary whose key-value pairs are to be inherited.
        child (dict): The dictionary to be updated with values from 'parent'.
        ignore_keys (Set[str]): A list of keys to be excluded from the update process.
        inplace (bool): If True, update 'child' dict inplace and do not return anything.

    Returns
    -------
        None: The function updates the 'child' dictionary in place and does not return anything.

    """
    if not inplace:
        child = child.copy()

    for key, val in parent.items():
        if key not in ignore_keys and (key not in child or not child[key]):
            child[key] = val

    if not inplace:
        return child


def reorder_trainer_cfg(cfg_dict: dict) -> dict:
    """Sort the OrderedDict based on the ordered_keys order
    and add any remaining keys from the original dict to the end.
    """
    ordered_keys = [
        "experiment_name", "save_dir", "git_hash", "seed", "reproducible",
        "device", "gpu_device", "use_amp", "torch_compile_model",
        "trainer", "model", "dataset", "dataloader",
        "optimizer", "lr_scheduler", "loss", "metrics"]

    # Sort the OrderedDict based on the key_order
    sorted_cfg_dict = OrderedDict(
        (k, cfg_dict[k]) for k in ordered_keys if k in cfg_dict)
    # add remaining keys to sorted_cfg_dict that were abset in ordered_keys
    sorted_cfg_dict.update(
        (k, v) for k, v in cfg_dict.items() if k not in ordered_keys)
    return sorted_cfg_dict


def recursively_flatten_config(config: DictConfig, parent_key: str = '', sep: str = '.') -> MutableMapping:
    """Recursively flattens a potentially nested OmegaConf DictConfig and returns
    a flat dictionary with the keys separated by sep.
    
    :param config: The DictConfig object to flatten.
    :param parent_key: The base key to prepend (used for recursion).
    :param sep: The separator for flattened keys.
    :return: A flattened dictionary.
    """
    items = []
    for key, value in config.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, DictConfig) and OmegaConf.is_dict(value):
            items.extend(recursively_flatten_config(
                value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


############################ file utils ############################


def find_latest_file_in_dir(dir_path: str, ext: str = "pth"):
    """Returns latest file with the ext from the dir_path directory
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


########################## data utils ##############################

def add_tfms(transform: Compose, tfms: Callable):
    """Add transforms to existing transforms"""
    transforms_list = transform.transforms
    transforms_list.append(tfms)
    return Compose(transforms_list)

####################################################################
