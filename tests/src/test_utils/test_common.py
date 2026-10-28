"""Tests for src.utils.common"""

import subprocess

from omegaconf import OmegaConf

from src.utils.common import BColors, get_git_revision_hash, recursively_flatten_config


def test_BColors():
    """Test BColors"""
    attrbs = ["HEADER", "OKBLUE", "OKCYAN", "OKGREEN", "WARN", "FAIL", "ENDC", "BOLD", "UNDERLINE"]
    for attrb in attrbs:
        assert hasattr(BColors, attrb)
    print(f"{BColors.WARN}Warning: Information.{BColors.ENDC}")


def test_get_git_revision_hash(mocker):
    """Assuming the test is run in a git-initialized folder"""
    expected_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    assert get_git_revision_hash() == expected_hash

    # Not a git repo, or git not installed (e.g. a slim Docker image)
    for error in (subprocess.CalledProcessError(1, "git"), FileNotFoundError("git")):
        mocker.patch("subprocess.check_output", side_effect=error)
        assert get_git_revision_hash() is None


def test_recursively_flatten_config():
    nested_dict = {"level1": {"level2": {"level3": "value3"}, "level2_value": "value2"}, "level1_value": "value1"}
    nested_cfg = OmegaConf.create(nested_dict)
    flat_dict = recursively_flatten_config(nested_cfg)
    expected_dict = {"level1.level2.level3": "value3", "level1.level2_value": "value2", "level1_value": "value1"}
    assert flat_dict == expected_dict
    nested_dict = OmegaConf.create(nested_dict)
    flat_dict = recursively_flatten_config(nested_dict, sep="=+=")
    expected_dict = {"level1=+=level2=+=level3": "value3", "level1=+=level2_value": "value2", "level1_value": "value1"}
    assert flat_dict == expected_dict
