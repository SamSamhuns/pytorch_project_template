"""Config loading for the entry scripts: their shared arguments, a YAML file with -o key=value overrides,
and the run dir each script saves to
"""

import argparse
import os
import os.path as osp
from datetime import datetime

from omegaconf import DictConfig, OmegaConf


def add_common_args(parser: argparse.ArgumentParser, run_prefix: str) -> None:
    """Add the arguments of every entry script: --cfg, -o and --id. Run ids default to run_prefix and the time"""
    parser.add_argument("--cfg", "--config", dest="config", required=True, help="YAML config file path")
    parser.add_argument(
        "-o",
        "--override",
        nargs="+",
        default=[],
        help="Override config values, e.g. -o trainer.epochs=2 optimizer.args.lr=1e-3 device=cpu",
    )
    parser.add_argument(
        "--id",
        "--run_id",
        dest="run_id",
        default=f"{run_prefix}_{datetime.now():%Y%m%d_%H%M%S}",
        help="Name of this run's dir in save_dir/experiment_name (default: %(default)s)",
    )


def load_config(path: str, overrides: list[str] | None = None) -> DictConfig:
    """Load the YAML config at path and apply overrides such as ["trainer.epochs=2", "optimizer.args.lr=1e-3"].
    The config is in struct mode, so a typo raises: overriding or reading a key the YAML doesn't have is an error
    """
    overrides = overrides or []
    for override in overrides:
        if "=" not in override:  # without "=", merge_with_dotlist would silently set the key to None
            raise ValueError(f"Invalid override format: {override}. Expected format: key=value")
    config = OmegaConf.load(path)
    OmegaConf.set_struct(config, True)
    config.merge_with_dotlist(overrides)  # parsed with OmegaConf's grammar, e.g. lr=1e-3 becomes a float
    return config


def make_run_dir(config: DictConfig, run_id: str) -> str:
    """Create the dir a run saves to, save_dir/experiment_name/run_id, with its logs and models dirs, and return it"""
    run_dir = osp.join(config.save_dir, config.experiment_name, run_id)
    os.makedirs(osp.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(osp.join(run_dir, "models"), exist_ok=True)
    return run_dir
