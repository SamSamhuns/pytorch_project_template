"""
Test configurations
"""
import os
import os.path as osp
import logging
import argparse
from typing import Callable, Tuple

import onnxruntime as ort
from PIL import Image
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset
from unittest.mock import MagicMock


PYTEST_TEMP_ROOT = "/tmp/pytest"
os.makedirs(PYTEST_TEMP_ROOT, exist_ok=True)


###### setup for augmentations test ######


@pytest.fixture
def sample_pilimage():
    """Create a dummy image"""
    def _sample_pilimage(h: int = 256, w: int = 256, c: int = 3):
        return Image.fromarray(np.uint8(np.random.rand(h, w, c) * 255))
    return _sample_pilimage


##########################################
###### setup for config parser test ######
SAMPLE_CFG_PATH = f"{PYTEST_TEMP_ROOT}/sample_config.json"


@pytest.fixture
def sample_config():
    return {
        "name": "test_model",
        "seed": 42,
        "gpu_device": None,
        "trainer": {
            "save_dir": PYTEST_TEMP_ROOT,
            "use_tensorboard": False
        },
        "optimizer": {
            "type": "SGD",
            "args": {
                "lr": 0.01
            }
        },
        "mode": "TRAIN"
    }


@pytest.fixture
def modifications():
    return {
        "trainer:save_dir": f"{PYTEST_TEMP_ROOT}/modified",
        "seed": 123
    }


@pytest.fixture
def mock_parser(scope="function"):
    """
    Generate a mock parser for testing based on train.py script
    """
    parser = argparse.ArgumentParser("Mock parser")
    parser.add_argument(
        '--cfg', '--config', type=str, dest="config", default=SAMPLE_CFG_PATH)
    parser.add_argument(
        '-r', '--resume', type=str, dest="resume", default=None)
    parser.add_argument(
        '--id', '--run_id', type=str, dest="run_id", default="train")
    parser.add_argument(
        '-v', '--verbose', action='store_true', dest="verbose", default=False)
    parser.add_argument(
        '-o', '--override', type=str, nargs='+', dest="override", default=None)
    return parser


@pytest.fixture
def override_args(scope="function"):
    """Returns a list of override args for testing"""
    return [
        {"flags": ['--dev', '--gpu_device'],
         "dest": "gpu_device",
         "help": "Config override arg: gpu_device list i.e. None or 0, 0 1, 0 1 2.",
         "nargs": "*",
         "type": int, "target": "gpu_device"},
        {"flags": ['--mode'],
         "dest": "mode",
         "help": "Running mode. (default: %(default)s)",
         "default": "TRAIN_TEST", "choices": ["TRAIN", "TRAIN_TEST", "TRAIN_TEST_FEATSELECT"],
         "type": str, "target": "mode"}
    ]


####################################
###### setup for loggers test ######


@pytest.fixture
def logger_test_dir():
    return f"{PYTEST_TEMP_ROOT}/test_logs"


@pytest.fixture
def gen_basic_config() -> Callable:
    """Fixture to generate data. It uses dynamic dimensions provided by test cases."""
    def _gen_basic_config(logger_name: str, logger_dir: str, **kwargs):
        cfg = {
            "logger_name": logger_name,
            "logger_dir": logger_dir,
            "file_fmt": '%(asctime)s %(levelname)-8s: %(message)s',
            "console_fmt": '%(message)s',
            "logger_level": logging.WARN,
            "file_level": logging.INFO,
            "console_level": logging.ERROR,
            "console_exc_keyword": "",
            "propagate": True
        }
        return cfg | kwargs
    return _gen_basic_config


#######################################
###### setup for dataloader test ######


class MockDataset(Dataset):
    """Mock a simple dataset"""

    def __init__(self, size, transform=None):
        self.size = size
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx % 10 == 0:  # return None for every 10th item
            return None
        return torch.tensor([idx] * 10, dtype=torch.float32)


@pytest.fixture
def mock_dataset() -> Dataset:
    """Return a mock dataset with size 100"""
    return MockDataset(100)


class MockWebDataset(Dataset):
    """Mock a simple webdataset"""

    def __init__(self, size=10):
        self.size = size
        self.data = torch.randn(size, 3, 24, 24)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.size


@pytest.fixture
def dummy_webdataset():
    """Return a mock webdataset"""
    return MockWebDataset()


####################################
###### setup for utils statistics & export_utils test ######


class Simple1DConvModel(nn.Module):
    """
    Inputs must be of shape [bsize, 3, 8]
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 16, 3, 1)
        self.fc1 = nn.Linear(96, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


@pytest.fixture
def simple_1d_conv_model():
    """Get simple model"""
    return Simple1DConvModel()


@pytest.fixture
def sample_tensor():
    """Sample tensor for Simple1DConvModel"""
    return torch.randn(32, 3, 8)


@pytest.fixture
def setup_model_and_logger(simple_1d_conv_model):
    """Get simple model and logger"""
    logger = MagicMock()
    return simple_1d_conv_model, logger


@pytest.fixture
def onnx_session(simple_1d_conv_model, sample_tensor):
    """Export pt model and return an ONNX Runtime session"""
    # Export the model to ONNX format temporarily
    export_path = PYTEST_TEMP_ROOT + "/temp_model.onnx"
    torch.onnx.export(
        simple_1d_conv_model, sample_tensor, export_path,
        export_params=True, opset_version=17,
        input_names=['input'], output_names=['output']
    )
    session = ort.InferenceSession(str(export_path))
    return session


####################################
##### Setup for datasets and trainers #####

##### Mocked Image data #####

NUM_CLS = 5
NUM_IMGS_P_CLS = 30


@pytest.fixture
def root_directory():
    return f"{PYTEST_TEMP_ROOT}/dataset/mock_imgs"


def _create_and_save_dummy_imgs(dir_path: str, n_cls: int = 5, n_imgs_p_cls: int = 30, size: Tuple[int, int] = (100, 100)) -> str:
    """
    Create and save dummy images to dir_path with image pixels separated by class.
    Parameters: 
        dir_path: str = path to root image directory
        n_cls: int = number of classes
        n_imgs_p_cls: int = number of images per class
        size: Tuple[int, int] = size in [width, height]
    Returns: str
    """
    os.makedirs(dir_path, exist_ok=True)
    classes = [f"class_{ci}" for ci in range(n_cls)]
    colorlist = [(ci * 255 // max(n_cls - 1, 1)) for ci in range(n_cls)]

    for ci, cls in enumerate(classes):
        cls_dir = osp.join(dir_path, cls)
        os.makedirs(cls_dir, exist_ok=True)

        for i in range(n_imgs_p_cls):
            img = Image.new('RGB', size, color=colorlist[ci])
            img.save(osp.join(cls_dir, f'{cls}_{i}.jpg'))


@pytest.fixture
def create_and_save_dummy_imgs() -> Callable:
    """Returns func to create and savae dummy imgs"""
    return _create_and_save_dummy_imgs


@pytest.fixture()
def mock_img_data_dir(root_directory, create_and_save_dummy_imgs) -> str:
    """Setup a directory with mocked image data"""
    create_and_save_dummy_imgs(
        root_directory, n_cls=NUM_CLS, n_imgs_p_cls=NUM_IMGS_P_CLS)
    return root_directory


#########################