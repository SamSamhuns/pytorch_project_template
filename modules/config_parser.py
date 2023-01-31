"""
CLI config parsing module
"""
import random
import warnings
from pathlib import Path
from operator import getitem
from datetime import datetime
from functools import partial, reduce
from typing import List, Dict, Union, Optional

import numpy as np
from modules.utils.util import read_json, write_json, validate_base_config_dict, MissingConfigError


class ConfigParser:
    """
    class to parse configuration json file.
    Handles hyperparameters for training, initializations of modules,
    checkpoint saving and logging module.
    :param config: Dict with configs & HPs to train. contents of `config/train_image_clsf.json` file for example.
    :param resume: String, path to the checkpoint being loaded.
    :param run_id: Unique Identifier for train & test. Used to save ckpts & training log. Timestamp is used as default
    """
    def __init__(self, config: dict,
                 run_id: Optional[str] = None,
                 resume: Optional[str] = None,
                 modification: Optional[dict] = None):
        # load config file and apply any modification
        config = _update_config(config, modification)
        # check base keys in config & raise warning if absent
        try:
            validate_base_config_dict(config)
        except MissingConfigError as err:
            warnings.warn(f"{err}")
        # set seeds
        seed = config["seed"]
        random.seed(seed)
        np.random.seed(seed)

        # set save_dir where trained model and log will be saved.
        save_dir = Path(config['trainer']['save_dir'])
        exper_name = config['name']
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%Y%m%d_%H_%M_%S')
        self._save_dir = save_dir / exper_name / run_id / 'models'
        self._log_dir = save_dir / exper_name / run_id / 'logs'

        # set tensorboard logging dirs
        if config["trainer"]["use_tensorboard"]:
            config["trainer"]["tensorboard_log_dir"] = str(
                save_dir / exper_name / run_id / config["trainer"]["tensorboard_log_dir"])

        # make directory for saving checkpoints and log.
        run_id = ''
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # save updated config file to the checkpoint dir
        write_json(config, self.save_dir / 'config.json')

        # set model info obj, config obj and resume chkpt
        self._config = config
        self.resume = resume

    @classmethod
    def from_args(cls, parser, options: Optional[List] = None):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        if options:
            # add optional overrride arguments to parser
            for opt in options:
                parser.add_argument(
                    *opt.flags, default=None, type=opt.type, dest=opt.dest, help=opt.help)
        if not isinstance(parser, tuple):
            args = parser.parse_args()

        resume = args.resume
        run_id = args.run_id
        cfg_fname = Path(args.config)
        config = read_json(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning or testing
            config.update(read_json(args.config))

        # parse custom cli options into dictionary
        modification = {opt.target: getattr(
            args, opt.dest) for opt in options} if options else None
        return cls(config, run_id, resume, modification)

    def init_obj(self, name, module, *args, **kwargs) -> object:
        """
        Finds a object handle with the 'name' given as 'type' in config, and
        returns the instance initialized with corresponding arguments given.
        'name' can also be list of keys and subkeys following access order in list
        to get the final 'type' and 'args' keys.

        `object = config.init_obj('name', module, a, b=1)` == `object = module.name(a, b=1)`
        `object = config.init_obj(['name', 'subname'], module, a, b=1)` == `object = module.name.subname(a, b=1)`
        """
        if isinstance(name, str):
            name = [name]
        module_name = _get_by_path(self, name + ["type"])
        module_args = dict(_get_by_path(self, name + ["args"]))
        assert all([k not in module_args for k in kwargs]
                   ), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs) -> callable:
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.
        'name' can also be list of keys and subkeys following access order in list
        to get the final 'type' and 'args' keys.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        if isinstance(name, str):
            name = [name]
        module_name = _get_by_path(self, name + ["type"])
        module_args = dict(_get_by_path(self, name + ["args"]))
        assert all([k not in module_args for k in kwargs]
                   ), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name: str):
        """Access items like ordinary dict."""
        return self.config[name]

    # set read-only attributes
    @property
    def config(self) -> dict:
        return self._config

    @property
    def save_dir(self) -> str:
        return self._save_dir

    @property
    def log_dir(self) -> str:
        return self._log_dir


##################################################################
# helper functions to update config dict with custom cli options #
##################################################################


def _update_config(config: Dict, modification: Optional[Dict[str, str]] = None) -> Dict:
    """
    Update config dict with param_path:value k:v pairs
    i.e. param_path : value = "optimizer;args;learning_rate" : 0.001
    """
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _set_by_path(tree: Dict, keys: List[str], value: str) -> None:
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree: Dict, keys: Union[str, List[str]]):
    """Access a nested object in tree by a key or sequence of keys."""
    return reduce(getitem, keys, tree)
