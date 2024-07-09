import os
import json
from pathlib import Path
from collections import OrderedDict

import pytest
from tests.conftest import PYTEST_TEMP_ROOT
from src.config_parser import ConfigParser, _update_config, _set_by_path, _get_by_path
from tests.conftest import SAMPLE_CFG_PATH


def test_config_parser_init(sample_config):
    config_parser = ConfigParser(config=sample_config)
    assert config_parser.config['seed'] == 42
    assert isinstance(config_parser.save_dir, Path)


def test_config_update_with_modifications(sample_config, modifications):
    updated_config = _update_config(sample_config, modifications)
    assert updated_config['trainer']['save_dir'] == f"{PYTEST_TEMP_ROOT}/modified"
    assert updated_config['seed'] == 123


def test_set_by_path(sample_config):
    _set_by_path(sample_config, "trainer:save_dir", "/new/save/dir")
    assert sample_config['trainer']['save_dir'] == "/new/save/dir"

    with pytest.raises(KeyError):
        _set_by_path(sample_config, "nonexistent:key", "value")


def test_get_by_path(sample_config):
    save_dir = _get_by_path(sample_config, ["trainer", "save_dir"])
    assert save_dir == PYTEST_TEMP_ROOT

    with pytest.raises(KeyError):
        _get_by_path(sample_config, ["trainer", "nonexistent"])


@pytest.mark.parametrize("modification_input, expected_output", [
    ({"optimizer:args:lr": 0.1}, {"optimizer:args": {"lr": 0.1}}),
    ({"seed": 100}, {"seed": 100})
])
def test_deep_update_config(sample_config, modification_input, expected_output):
    updated_config = _update_config(sample_config, modification_input)
    for key, value in expected_output.items():
        keys = key.split(':')
        assert _get_by_path(updated_config, keys) == value


def test_from_args(monkeypatch, mock_parser, sample_config, override_args):
    """Test ConfigParser init with override args"""
    os.makedirs(PYTEST_TEMP_ROOT, exist_ok=True)
    with open(SAMPLE_CFG_PATH, 'w', encoding="utf-8") as f:
        json.dump(sample_config, f)

    # modify sample config to match the override args
    sample_config["mode"] = "TRAIN_TEST"
    sample_config["trainer"] = OrderedDict(sample_config["trainer"])
    sample_config = OrderedDict(sample_config)

    monkeypatch.setattr('sys.argv', ['train.py', '--cfg', SAMPLE_CFG_PATH])
    config_parser = ConfigParser.from_args(mock_parser, override_args)

    assert "git_hash" in config_parser.config
    del config_parser.config["git_hash"]
    assert config_parser.config == sample_config
