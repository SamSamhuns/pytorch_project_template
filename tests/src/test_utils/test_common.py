"""
Tests for src.utils.common
"""
import pytest
import errno
import socket
import subprocess
from omegaconf import OmegaConf

from tests.conftest import PYTEST_TEMP_ROOT
from src.utils.common import (
    BColors, round_to_nearest_divisor, can_be_conv_to_float, try_bool, try_null,
    capture_output, is_port_in_use, get_git_revision_hash, stable_sort,
    inherit_missing_dict_params, reorder_trainer_cfg,
    recursively_flatten_config)


def test_BColors():
    """Test BColors"""
    attrbs = ["HEADER", "OKBLUE", "OKCYAN", "OKGREEN", "WARN",
              "FAIL", "ENDC", "BOLD", "UNDERLINE"]
    for attrb in attrbs:
        assert hasattr(BColors, attrb)
    print(f"{BColors.WARN}Warning: Information.{BColors.ENDC}")


def test_try_bool():
    assert try_bool("true") is True
    assert try_bool("false") is False
    assert try_bool("TRUE") is True
    assert try_bool("FALSE") is False
    with pytest.raises(ValueError):
        try_bool("yes")
    with pytest.raises(ValueError):
        try_bool("no")
    with pytest.raises(ValueError):
        try_bool("yes1")


def test_try_null():
    assert try_null("null") is None
    assert try_null("Null") is None
    assert try_null("NULL") is None
    assert try_null("None") is None
    assert try_null("none") is None
    with pytest.raises(ValueError):
        try_null("none1")


def test_can_be_conv_to_float():
    assert can_be_conv_to_float(1)
    assert can_be_conv_to_float(1.0)
    assert can_be_conv_to_float('1.0')
    assert can_be_conv_to_float('1')
    assert not can_be_conv_to_float('1.0.0')
    assert not can_be_conv_to_float('1.0.0.0')
    assert not can_be_conv_to_float('apple')


def test_get_git_revision_hash():
    """Assuming the test is run in a git-initialized folder"""
    expected_hash = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    assert get_git_revision_hash() == expected_hash

    # Mock subprocess to simulate a non-git repo situation
    def raiser(*args, **kwargs):
        raise subprocess.CalledProcessError(1, 'git')
    subprocess.check_output = raiser
    assert get_git_revision_hash() is None


def test_capture_stdout(test_str="Test string xxx"):
    """Check if stdout is captured"""
    def _print(x: str):
        print(test_str)
        print(x)
    output = capture_output(_print, "param1")
    assert output == test_str + "\nparam1\n"


def test_is_port_in_use():
    """This test assumes no service is running on port 50000"""
    assert not is_port_in_use(50000)

    # Mock socket to simulate a port in use situation
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 50001))
        s.listen(1)
        assert is_port_in_use(50001)
    except socket.error as e:
        if e.errno == errno.EADDRINUSE:
            print("Port is already in use")
        else:
            # something else raised the socket.error exception
            print(e)
        assert False
    s.close()


def test_round_to_nearest_divisor():
    assert round_to_nearest_divisor(10, 3) == 9
    assert round_to_nearest_divisor(14, 5) == 15
    assert round_to_nearest_divisor(6, 6) == 6
    assert round_to_nearest_divisor(0, 5) == 0


@pytest.mark.parametrize("input_array, expected_output", [
    (['1', '10', '2'], ['1', '2', '10']),
    ([7.0, '1', '2', '10', 5, 'xyz', 'a'], ['1', '2', 5, 7.0, '10', 'a', 'xyz']),
    (['2', 'a', 'c', 'b', '1'], ['1', '2', 'a', 'b', 'c'])
])
def test_stable_sort(input_array, expected_output):
    assert stable_sort(input_array) == expected_output


@pytest.mark.parametrize("parent, child, ignore_keys, expected_result", [
    ({"a": 1, "b": 2}, {}, set(), {"a": 1, "b": 2}),
    ({"a": 1, "b": 2}, {"b": 3}, set(), {"a": 1, "b": 3}),  # child retains its value
    ({"a": 1, "b": 2}, {"b": 3}, {"b"}, {"b": 3, "a": 1})  # b is ignored
])
def test_inherit_missing_dict_params(parent, child, ignore_keys, expected_result):
    inherit_missing_dict_params(parent, child, ignore_keys)
    assert child == expected_result


def test_reorder_trainer_cfg():
    cfg = {
        "model": "resnet",
        "loss": "mse",
        "optimizer": "adam",
        "name": "experiment1",
        "unexpected_key": "value"
    }
    sorted_cfg = reorder_trainer_cfg(cfg)
    actual_keys_order = list(sorted_cfg.keys())
    assert actual_keys_order[:4] == ["name", "model", "optimizer", "loss"]
    assert "unexpected_key" in actual_keys_order


def test_recursively_flatten_config():
    nested_dict = {
        "level1": {
            "level2": {
                "level3": "value3"
            },
            "level2_value": "value2"
        },
        "level1_value": "value1"
    }
    nested_cfg = OmegaConf.create(nested_dict)
    flat_dict = recursively_flatten_config(nested_cfg)
    expected_dict = {
        "level1.level2.level3": "value3",
        "level1.level2_value": "value2",
        "level1_value": "value1"
    }
    assert flat_dict == expected_dict
    nested_dict = OmegaConf.create(nested_dict)
    flat_dict = recursively_flatten_config(nested_dict, sep="=+=")
    expected_dict = {
        "level1=+=level2=+=level3": "value3",
        "level1=+=level2_value": "value2",
        "level1_value": "value1"
    }
    assert flat_dict == expected_dict
