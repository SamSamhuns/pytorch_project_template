"""CLI config parsing module with OmegaConf and YAML support."""
import argparse
import os
import os.path as osp
import random
from datetime import datetime

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from .utils.common import BColors, get_git_revision_hash


def parse_omegaconf_primitive(val_str):
    """Parse a string representation of a omegaconf value (int, float, str, bool, ${oc.env:HOME}) and returns the corresponding Python type.

    Args:
        val_str: String representation of the value.

    Returns:
        Parsed value as a Python type (int, float, str, bool).

    """
    wrapped = OmegaConf.create({"_val_": yaml.safe_load(val_str)})
    return OmegaConf.to_container(wrapped, resolve=True)["_val_"]


class CustomDictConfig(DictConfig):
    """A wrapper around OmegaConf's DictConfig to extend its functionality.

    Handles additional tasks like setting up directories, logging, and
    applying runtime modifications.

    Args:
        config: DictConfig object with configurations.
        run_id: Unique Identifier for train & test. Used to save ckpts & training log.
        modification: Additional key-value pairs to override in config.

    """

    def __init__(self,
                 config: DictConfig,
                 run_id: str | None = None,
                 verbose: bool = False,
                 modification: dict | None = None):
        super().__init__(config)

        # Apply any modifications to the configuration
        if modification:
            # Removes keys that have None as values
            modification = {k: v for k, v in modification.items() if v is not None}
            for k, v in modification.items():
                OmegaConf.update(self, k, v, merge=True)
        # any cfgs should be received from self not config now

        # set seeds if present
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed + 1)
            torch.manual_seed(self.seed + 2)
            if torch.cuda.is_available() and self.device == "cuda":
                torch.cuda.manual_seed(self.seed + 3)

        # Configure reproducibility settings
        if self.reproducible:
            print(f"{BColors.WARN}Warning: " +
                  "Setting torch.backends.cudnn.deterministic to True. " +
                  f"This may slow down GPU training.{BColors.ENDC}"
                  )
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # If run_id is None, use timestamp as default run-id
        if run_id is None:
            run_id = datetime.now().strftime(r"%Y%m%d_%H%M%S")
        self.run_id = run_id
        self.verbose = verbose
        self.git_hash = get_git_revision_hash()

        # Set directories for saving logs, metrics and models
        save_root = osp.join(self.save_dir, self.experiment_name, run_id)
        _logs_dir = osp.join(save_root, "logs")
        _metrics_dir = osp.join(save_root, "metrics")
        _models_dir = osp.join(save_root, "models")

        # Create necessary directories
        os.makedirs(_logs_dir, exist_ok=True)
        os.makedirs(_metrics_dir, exist_ok=True)
        os.makedirs(_models_dir, exist_ok=True)

        # Save the updated config to the save_root directory
        OmegaConf.save(self, osp.join(save_root, "config.yaml"))
        # assign updated logs, metrics and model dir after saving config
        self.logs_dir = _logs_dir
        self.metrics_dir = _metrics_dir
        self.models_dir = _models_dir
        if self.trainer.use_tensorboard:
            self.tboard_log_dir = osp.join(save_root, "tf_logs")

    @classmethod
    def from_args(cls,
                  args: argparse.Namespace,
                  modification: dict | None = None,
                  add_all_args: bool = True):
        """Initialize this class from CLI arguments. Used in train, test.

        Args:
            args: Parsed CLI arguments.
            modification: Key-value pair to override in config.
                          Can have nested structure separated by periods(.)
                          e.g. {"key1":"val1", "key2.sub_key2":"val2"}
            add_all_args: Add all args to modification 
                          that are not alr present as top-level keys.

        """
        modification = {} if not modification else modification
        # Add all args to modification from args
        if add_all_args:
            # only check top-level keys
            mod_keys = {k.rsplit('.')[0] for k in modification}
            for arg, value in vars(args).items():
                # add new keys not present in orig yaml config
                if arg not in mod_keys and arg not in {"override"}:
                    modification[arg] = value

        # Load configuration from YAML
        config = OmegaConf.load(args.config)
        # Apply dotlist overrides (-o)
        if args.override:
            OmegaConf.set_struct(config, True)  # Enable strict mode to disallow unknown keys
            for override in args.override:
                if '=' not in override:
                    raise ValueError(f"Invalid override format: {override}. Expected format: key=value")
                key, val_str = override.split("=", 1)
                OmegaConf.update(config, key, parse_omegaconf_primitive(val_str))
            OmegaConf.set_struct(config, False)  # Disable strict mode to allow runtime modifications later

        return cls(config, args.run_id, args.verbose, modification)

    def __str__(self):
        return OmegaConf.to_yaml(self)
