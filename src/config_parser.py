"""
CLI config parsing module
"""
import random
import warnings
from pathlib import Path
from datetime import datetime
from operator import getitem
from functools import reduce
from typing import List, Dict, Union, Optional

import numpy as np
from .utils.common import BColors, try_bool, try_null, read_json, write_json, get_git_revision_hash


class ConfigParser:
    """
    class to parse configuration json file.
    Handles hyperparameters for training, initializations of modules,
    checkpoint saving and logging module.
    Args:
        config: Dict with configs & HPs to train. contents of `config/classifier_cpu_config.json` file for example.
        run_id: Unique Identifier for train & test. Used to save ckpts & training log. Timestamp is used as default
        resume: String, path to the checkpoint being loaded.
        modification: additional key-val args to be added to config
    """

    def __init__(self, config: dict,
                 run_id: Optional[str] = None,
                 resume: Optional[str] = None,
                 verbose: bool = False,
                 modification: Optional[dict] = None):
        # load config file and apply any modification
        config = _update_config(config, modification)
        # set seeds
        seed = config["seed"]
        random.seed(seed)
        np.random.seed(seed)

        # set save_dir where trained model and log will be saved.
        save_dir = Path(config['trainer']['save_dir'])
        exper_name = config['name']
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%Y%m%d_%H%M%S')
        self._save_dir = save_dir / exper_name / run_id / 'models'
        self._log_dir = save_dir / exper_name / run_id / 'logs'

        # make directory for saving checkpoints and log.
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if config["trainer"]["use_tensorboard"]:
            self._tboard_log_dir = save_dir / exper_name / run_id / "tf_logs"

        # try to get git hash if inside git repo, otherwise use None
        config["git_hash"] = get_git_revision_hash()

        # save updated config file to the checkpoint dir
        write_json(config, self.save_dir / 'config.json')

        # set model info obj, config obj and resume chkpt
        self._config = config
        self.resume = resume
        self.run_id = run_id
        self.verbose = verbose

    @classmethod
    def from_args(cls, parser, options: Optional[List[dict]] = None):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        if options:
            # add optional overrride arguments to parser
            for opt_dict in options:
                # unpack opt_dict ignoring key 'target'
                parser.add_argument(
                    *opt_dict["flags"],
                    **{k: opt_dict[k] for k in opt_dict if k not in {"flags", "target"}})
        if not isinstance(parser, tuple):
            args = parser.parse_args()
        # read config_dict from JSON cfg file
        config_dict = read_json(args.config)
        # parse custom cli options into mod dictionary
        modification = {opt["target"]: getattr(
            args, opt["dest"]) for opt in options} if options else {}

        # parse override options into mod dictionary
        if args.override:
            for opt in args.override:
                parts = opt.split(':')
                key, val = ':'.join(parts[:-1]), parts[-1]
                # try to cast val as int or float or bool or null else use str as-is
                for cast in (int, float, try_bool, try_null):
                    try:
                        val = cast(val)
                        break  # Exit loop if conversion successful
                    except ValueError:
                        continue
                modification[key] = val
        return cls(config_dict, args.run_id, args.resume, args.verbose, modification)

    def __getitem__(self, key: str):
        """Access items like ordinary dict."""
        return self.config[key]

    def __str__(self):
        return str(self._config)

    def __iter__(self):
        for k, v in self._config.items():
            yield k, v

    # set read-only attributes
    @property
    def config(self) -> dict:
        """Get config dict."""
        return self._config

    @property
    def save_dir(self) -> Path:
        """Get model save dir as posixpath."""
        return self._save_dir

    @property
    def log_dir(self) -> Path:
        """Get model log dir as posixpath."""
        return self._log_dir

    @property
    def tboard_log_dir(self) -> Path:
        """Get tboard log dir as posixpath."""
        return self._tboard_log_dir


##################################################################
# helper functions to update config dict with custom cli options #
##################################################################


def _update_config(config: Dict, modification: Optional[Dict[str, str]] = None) -> Dict:
    """
    Updates a configuration dictionary in a deep manner based on hierarchical key paths specified
    in the modification dictionary. Each key in the modification dictionary specifies a path through
    the configuration hierarchy, separated by colons (':'), and the corresponding value is set at that
    path.

    Args:
        config: The original configuration dictionary to be updated.
        modification: An optional dictionary with keys as hierarchical key paths (separated by colons) 
                      indicating where in the config to set the values, and values being the new data 
                      to set.

    Returns:
        Dict: The updated configuration dictionary.

    Example:
        To update the learning rate in an optimizer configuration, use the modification:
        {"optimizer:args:learning_rate": 0.001}
    """
    if not modification:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _set_by_path(tree: Dict, keys: List[str], value: str) -> None:
    """
    Sets a value in a nested dictionary based on a sequence of keys separated by colons.
    If the entire path of keys does not exist, a KeyError is raised.

    Args:
        tree: The dictionary to be modified.
        keys: A string representing the hierarchical path to a specific configuration item, 
              separated by colons (':').
        value: The value to set at the specified path.

    Raises:
        KeyError: If the final key does not exist in the dictionary at the end of the path.
    """
    keys = keys.split(':')
    # currently does not raise KeyError if key does not exist in config
    if len(keys) == 1 and keys[-1] not in tree:
        with warnings.catch_warnings():
            warnings.simplefilter('always')  # enable all warnings
            msg = f"{BColors.WARN}Key '{keys[-1]}' missing in config. Consider adding key to JSON config first.{BColors.ENDC}"
            warnings.warn(msg)
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree: Dict, keys: Union[str, List[str]]):
    """
    Retrieves a value from a nested dictionary using a specified path of keys, which can be
    a single key or a sequence of keys.

    Args:
        tree: The dictionary from which to retrieve the value.
        keys: A key or a list of keys representing the path to the desired value.

    Returns:
        The value located at the specified path.
    """
    return reduce(getitem, keys, tree)
