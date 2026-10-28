"""Contains utility functions meant to be used internally & externally"""

import logging
import random
import subprocess

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_git_revision_hash() -> str:
    """Get the git hash of the current commit. Returns None if run from a non-git init repo"""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except (subprocess.CalledProcessError, OSError) as excep:  # OSError: git is not installed
        logger.warning("Couldn't get the git hash of the current repo, so it isn't recorded: %s", excep)
    return None


def get_device(device: str) -> torch.device:
    """The device to run on, e.g. cpu, cuda or cuda:1. Falls back to the CPU, with a warning, without CUDA"""
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is not available, so the program will run on CPU")
        device = "cpu"
    return torch.device(device)


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch. torch.manual_seed also seeds every CUDA device"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def identity(x):
    return x
