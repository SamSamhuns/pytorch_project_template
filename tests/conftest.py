"""Test configurations"""

import os
import os.path as osp
import shutil
from collections.abc import Callable

import numpy as np
import pytest
from PIL import Image
from torch import nn

from scripts.convert_dataset_to_tar import write_shards

PYTEST_TEMP_ROOT = "/tmp/pytest"
os.makedirs(PYTEST_TEMP_ROOT, exist_ok=True)


###### setup for augmentations test ######


@pytest.fixture
def sample_pilimage():
    """Create a dummy image"""

    def _sample_pilimage(h: int = 256, w: int = 256, c: int = 3):
        return Image.fromarray(np.uint8(np.random.rand(h, w, c) * 255))

    return _sample_pilimage


####################################
###### setup for utils statistics & export_utils test ######


class Simple2DConvModel(nn.Module):
    """Inputs must be of shape [bsize, 3, H, W] where H and W are the height and width of the input images."""

    def __init__(self, height=100, width=100):
        super().__init__()
        # Assuming input channels are 3, the output channels are 16, kernel size is (3, 3) and stride is 1
        self.conv1 = nn.Conv2d(3, 16, (3, 3), stride=1)
        # Calculate the output dimensions after the convolution
        # (dimension - kernel_size + 2*padding) / stride + 1
        # Assuming padding is 0 (default) and stride is 1:
        output_height = height - 3 + 1
        output_width = width - 3 + 1

        # Number of features for the linear layer
        num_features = 16 * output_height * output_width

        self.fc1 = nn.Linear(num_features, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)  # Flatten the features into a vector
        x = self.fc1(x)
        return x


@pytest.fixture
def simple_2d_conv_model():
    """Get simple model"""
    return Simple2DConvModel()


####################################
##### Setup for datasets and trainers #####

##### Mocked Image data #####

NUM_CLS = 5
NUM_IMGS_P_CLS = 30


@pytest.fixture
def root_directory():
    return f"{PYTEST_TEMP_ROOT}/dataset/mock_imgs"


def _create_and_save_dummy_imgs(
    dir_path: str, n_cls: int = 5, n_imgs_p_cls: int = 30, size: tuple[int, int] = (100, 100)
) -> str:
    """Create and save dummy images to dir_path with image pixels separated by class.

    Parameters
    ----------
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
            img = Image.new("RGB", size, color=colorlist[ci])
            img.save(osp.join(cls_dir, f"{cls}_{i:03d}.jpg"))


@pytest.fixture
def create_and_save_dummy_imgs() -> Callable:
    """Returns func to create and savae dummy imgs"""
    return _create_and_save_dummy_imgs


@pytest.fixture()
def dump_mock_img_data_dir(root_directory, create_and_save_dummy_imgs) -> str:
    """Setup a directory with mocked image data"""
    img_dir = osp.join(root_directory, "imgs")
    create_and_save_dummy_imgs(img_dir, n_cls=NUM_CLS, n_imgs_p_cls=NUM_IMGS_P_CLS)
    return img_dir


@pytest.fixture()
def mock_webdataset_path(root_directory, dump_mock_img_data_dir) -> str:
    """The mock images as WebDataset shards of 50 images each; returns the dir of the shards and info.json"""
    shard_dir = osp.join(root_directory, "webdataset")
    shutil.rmtree(shard_dir, ignore_errors=True)  # write_shards needs a new dir
    write_shards(dump_mock_img_data_dir, shard_dir, maxcount=50)
    return shard_dir


#########################
