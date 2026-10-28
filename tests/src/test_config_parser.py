"""Tests for src.config_parser"""

import argparse
import os.path as osp

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError

from src.config_parser import add_common_args, load_config, make_run_dir


@pytest.fixture
def config_path(tmp_path) -> str:
    config = {"experiment_name": "exp", "save_dir": str(tmp_path / "runs"), "optimizer": {"args": {"lr": 0.01}}}
    OmegaConf.save(OmegaConf.create(config), tmp_path / "config.yaml")
    return str(tmp_path / "config.yaml")


def test_override_parses_scientific_notation(config_path):
    """-o optimizer.args.lr=1e-3 must give the float 0.001, not the string '1e-3'"""
    assert load_config(config_path, ["optimizer.args.lr=1e-3"]).optimizer.args.lr == 0.001


def test_typos_raise(config_path):
    """Overriding or reading a key the YAML doesn't have is an error, not a silent new key or None"""
    with pytest.raises(ConfigAttributeError):
        load_config(config_path, ["optimizer.args.lrr=0.1"])
    with pytest.raises(ConfigAttributeError):
        _ = load_config(config_path).optimizer.momentum


def test_override_needs_a_value(config_path):
    with pytest.raises(ValueError, match="key=value"):
        load_config(config_path, ["optimizer.args.lr"])


def test_make_run_dir(config_path):
    run_dir = make_run_dir(load_config(config_path), "train_1")
    assert run_dir.endswith(osp.join("runs", "exp", "train_1"))
    assert osp.isdir(osp.join(run_dir, "logs"))
    assert osp.isdir(osp.join(run_dir, "models"))


def test_common_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser, run_prefix="train")
    args = parser.parse_args(["--cfg", "configs/mnist_config.yaml", "-o", "trainer.epochs=2", "device=cpu"])
    assert args.config == "configs/mnist_config.yaml"
    assert args.override == ["trainer.epochs=2", "device=cpu"]
    assert args.run_id.startswith("train_")
