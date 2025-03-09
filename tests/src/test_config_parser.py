import os
import tempfile
from omegaconf import OmegaConf
from unittest.mock import patch
from src.config_parser import CustomDictConfig, apply_modifications, parse_and_cast_kv_overrides


def test_apply_modifications():
    config = OmegaConf.create({"level1": {"level2": {"key": "value"}}})
    modifications = {"level1:level2:key": "new_value",
                     "level1:level3": "added_value"}
    apply_modifications(config, modifications)

    assert config.level1.level2.key == "new_value"
    assert config.level1.level3 == "added_value"


def test_parse_and_cast_kv_overrides():
    overrides = ["key1:123", "key2:45.6", "key3:true", "key4:null"]
    modifications = {}

    parse_and_cast_kv_overrides(overrides, modifications)

    assert modifications["key1"] == 123
    assert modifications["key2"] == 45.6
    assert modifications["key3"] is True
    assert modifications["key4"] is None


def test_CustomDictConfig_initialization():
    config_dict = {
        "experiment_name": "test_run",
        "save_dir": "/tmp",
        "seed": 42,
        "reproducible": True,
        "device": "cpu",
        "trainer": {"use_tensorboard": True}
    }
    config = OmegaConf.create(config_dict)

    with tempfile.TemporaryDirectory() as tmpdir:
        config.save_dir = tmpdir
        cfg = CustomDictConfig(config)

        assert cfg.save_dir == os.path.join(tmpdir)
        assert os.path.exists(cfg["save_dir"])
        assert os.path.exists(cfg["models_dir"])
        assert os.path.exists(cfg["logs_dir"])
        assert os.path.exists(cfg["metrics_dir"])
        assert os.path.isfile(os.path.join(
            tmpdir, "test_run", cfg.run_id, "config.yaml"))


def test_CustomDictConfig_from_args():
    config_dict = {
        "experiment_name": "test_run",
        "save_dir": "/tmp",
        "seed": 42,
        "reproducible": True,
        "device": "cpu",
        "trainer": {"use_tensorboard": True}
    }
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_config:
        OmegaConf.save(OmegaConf.create(config_dict), temp_config.name)
        temp_config_path = temp_config.name

    class MockArgs:
        config = temp_config_path
        run_id = None
        verbose = True
        override = ["name:new_name"]

    args = MockArgs()
    parser = CustomDictConfig.from_args(args)

    assert parser.name == "new_name"
    assert parser.verbose is True

    os.remove(temp_config_path)


def test_CustomDictConfig_get_set_item():
    config_dict = {
        "experiment_name": "test_run",
        "save_dir": "/tmp",
        "seed": 42,
        "reproducible": True,
        "device": "cpu",
        "trainer": {"use_tensorboard": True}
    }
    config = OmegaConf.create(config_dict)
    parser = CustomDictConfig(config)

    parser["experiment_name"] = "updated_name"
    assert parser["experiment_name"] == "updated_name"

    parser["new_key"] = "new_value"
    assert parser["new_key"] == "new_value"


def test_CustomDictConfig_str():
    config_dict = {
        "experiment_name": "test_run",
        "save_dir": "/tmp",
        "seed": 42,
        "reproducible": True,
        "device": "cpu",
        "trainer": {"use_tensorboard": True}
    }
    config = OmegaConf.create(config_dict)
    parser = CustomDictConfig(config)

    config_str = str(parser)
    assert "save_dir" in config_str
    assert "experiment_name" in config_str


def test_CustomDictConfig_iter():
    config_dict = {
        "experiment_name": "rand",
        "save_dir": "/tmp",
        "run_id": "bar",
        "key1": "value1",
        "key2": "value2",
        "seed": 42,
        "reproducible": True,
        "device": "cpu",
        "trainer": {"use_tensorboard": True}
    }
    config = OmegaConf.create(config_dict)
    parser = CustomDictConfig(config, run_id="bar")
    config_dict.pop("save_dir")

    items = {
        k: v for k, v in parser.items()
        if k not in {"git_hash", "save_dir", "models_dir", "logs_dir", "metrics_dir", "verbose", "tboard_log_dir"}}

    assert items == config_dict


@patch("src.config_parser.get_git_revision_hash", return_value="fake_hash")
def test_git_hash_in_config(mock_get_git_revision_hash):
    config_dict = {
        "experiment_name": "test_run",
        "save_dir": "/tmp",
        "seed": 42,
        "reproducible": True,
        "device": "cpu",
        "trainer": {"use_tensorboard": True}
    }
    config = OmegaConf.create(config_dict)
    parser = CustomDictConfig(config)

    assert parser.git_hash == "fake_hash"

    mock_get_git_revision_hash.assert_called_once()
