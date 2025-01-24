"""
CLI config parsing module with OmegaConf and YAML support
"""
import os
import os.path as osp
import argparse
import random
from datetime import datetime
from typing import List, Optional

import numpy as np
from omegaconf import OmegaConf, DictConfig
from .utils.common import get_git_revision_hash, try_bool, try_null


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
    Parses a list of key-val override strings and casts values to appropriate types inplace.
    Args:
        override (List[str]): List of strings in the format "key:value" or "key:child1:child2:....:val".
                 Bool should be passed as true & None should be passed as null
    """
    for opt in override:
        key, val = opt.rsplit(":", 1)
        # Attempt to cast the value to an appropriate type
        for cast in (int, float, try_bool, try_null):
            try:
                val = cast(val)
                break
            except (ValueError, TypeError):
                continue
        modification[key] = val


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
            modification = {k: v for k, v in modification.items() if v}
            apply_modifications(self, modification)

        # set seeds
        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        # If run_id is None, use timestamp as default run-id
        if run_id is None:
            run_id = datetime.now().strftime(r"%Y%m%d_%H%M%S")
        self.run_id = run_id
        self.verbose = verbose
        self.git_hash = get_git_revision_hash()

        # Set directories for saving logs and models
        _log_dir = osp.join(config.save_dir, config.name, run_id, "logs")
        _save_dir = osp.join(config.save_dir, config.name, run_id, "models")

        # Create necessary directories
        os.makedirs(_save_dir, exist_ok=True)
        os.makedirs(_log_dir, exist_ok=True)

        # Save the updated config to the save directory
        OmegaConf.save(self, osp.join(_save_dir, "config.yaml"))
        # assign updated logs and save dir after saving config
        self.log_dir = _log_dir
        self.save_dir = _save_dir
        if config.trainer.use_tensorboard:
            self.tboard_log_dir = osp.join(
                config.save_dir, config.name, run_id, "tf_logs")

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
                if arg not in mod_keys:
                    modification[arg] = value
        # Override configuration parameters if args.override is provided
        if args.override:
            parse_and_cast_kv_overrides(args.override, modification)

        # Load configuration from YAML
        config = OmegaConf.load(args.config)
        return cls(config, args.run_id, args.verbose, modification)

    def __str__(self):
        return OmegaConf.to_yaml(self)
