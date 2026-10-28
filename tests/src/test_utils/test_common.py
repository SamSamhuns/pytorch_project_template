"""Tests for src.utils.common"""

import random
import subprocess

import numpy as np
import torch

from src.utils.common import get_git_revision_hash, seed_everything


def test_seed_everything():
    """The same seed gives the same python, numpy and torch draws"""
    draws = []
    for _ in range(2):
        seed_everything(7)
        draws.append((random.random(), np.random.rand(), torch.rand(1).item()))
    assert draws[0] == draws[1]


def test_get_git_revision_hash(mocker):
    """Assuming the test is run in a git-initialized folder"""
    expected_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    assert get_git_revision_hash() == expected_hash

    # Not a git repo, or git not installed (e.g. a slim Docker image)
    for error in (subprocess.CalledProcessError(1, "git"), FileNotFoundError("git")):
        mocker.patch("subprocess.check_output", side_effect=error)
        assert get_git_revision_hash() is None
