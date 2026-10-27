"""Contains utility functions meant to be used internally & externally
"""
import subprocess
from collections.abc import MutableMapping

from omegaconf import DictConfig, OmegaConf


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


def get_git_revision_hash() -> str:
    """Get the git hash of the current commit. Returns None if run from a non-git init repo"""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except (subprocess.CalledProcessError, OSError) as excep:  # OSError: git is not installed
        print(excep, "Couldn't get git hash of the current repo. Returning None")
    return None


def identity(x):
    return x


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
