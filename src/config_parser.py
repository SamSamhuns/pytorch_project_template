"""
CLI config parsing module with OmegaConf and YAML support
"""
import os
import os.path as osp
import argparse
import random
from datetime import datetime
from typing import List, Optional, Any

import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig
from .utils.common import get_git_revision_hash, try_bool, try_null, BColors


def apply_modifications(config: DictConfig, modification: dict) -> None:
    """
    Applies modifications to a nested DictConfig object using colon-separated keys inplace.
    Args:
        config (DictConfig): Original configuration object.
        modification (dict): Dictionary with colon-separated keys representing the hierarchy and values to override.
    """
    for key, value in modification.items():
        path = key.split(":")
        node = config
        for part in path[:-1]:  # Traverse to the parent node
            if part not in node:
                node[part] = {}  # Create nested structure if missing
            node = node[part]
        node[path[-1]] = value


def parse_and_cast_kv_overrides(override: List[str], modification: dict) -> None:
    """
    Parses a list of key-value override strings and casts values to appropriate types in place.
    Supports overriding lists using comma-separated values.

    Args:
        override (List[str]): List of strings in the format "key:value" or "key:sub_key1:sub_key2:val".
                              Lists can also be passed as values with "key:val1,val2,val3".
        modification (dict): Dictionary to store parsed key-value pairs.
    """
    def cast_value(val) -> Any:
        """Attempts to cast a value to int, float, bool, None, or leaves it as a string."""
        for cast in (int, float, try_bool, try_null):
            try:
                return cast(val)
            except (ValueError, TypeError):
                continue
        return val  # Fallback to string if no other type matches

    for opt in override:
        # split at the last colon, i.e. key:subkey:val -> key:subkey, val
        key, val = opt.rsplit(":", 1)
        if "," in val:  # If it's a list
            modification[key] = [cast_value(v) for v in val.split(",")]
        else:
            modification[key] = cast_value(val)


class CustomDictConfig(DictConfig):
    """
    A wrapper around OmegaConf's DictConfig to extend its functionality.
    Handles additional tasks like setting up directories, logging, and
    applying runtime modifications.
    Args:
        config: DictConfig object with configurations.
        run_id: Unique Identifier for train & test. Used to save ckpts & training log.
        modification: Additional key-value pairs to override in config.
    """

    def __init__(self,
                 config: DictConfig,
                 run_id: Optional[str] = None,
                 verbose: bool = False,
                 modification: Optional[dict] = None):
        super().__init__(config)

        # Apply any modifications to the configuration
        if modification:
            # Removes keys that have None as values
            modification = {k: v for k, v in modification.items() if v is not None}
            apply_modifications(self, modification)
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
                  modification: Optional[dict] = None,
                  add_all_args: bool = True):
        """
        Initialize this class from CLI arguments. Used in train, test.
        Args:
            args: Parsed CLI arguments.
            modification: Key-value pair to override in config.
                          Can have nested structure separated by colons.
                          e.g. ["key1:val1", "key2:sub_key2:val2"]
            add_all_args: Add all args to modification 
                          that are not alr present as top-level keys.
        """
        modification = {} if not modification else modification
        # Add all args to modification from args
        if add_all_args:
            # only check top-level keys
            mod_keys = {k.rsplit(':')[0] for k in modification}
            for arg, value in vars(args).items():
                # add new keys not present in orig yaml config
                if arg not in mod_keys and arg not in {"override"}:
                    modification[arg] = value
        # Override configuration parameters if args.override is provided
        if args.override:
            parse_and_cast_kv_overrides(args.override, modification)

        # Load configuration from YAML
        config = OmegaConf.load(args.config)
        return cls(config, args.run_id, args.verbose, modification)

    def __str__(self):
        return OmegaConf.to_yaml(self)
