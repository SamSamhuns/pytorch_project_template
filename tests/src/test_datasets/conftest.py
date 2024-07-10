import os.path as osp

import torch
import pytest
from torchvision import transforms
from src.datasets.base_dataset import ImageFolderDataset
from src.datasets import ClassifierDataset

# ######### Fixtures for BaseDataset #########

@pytest.fixture()
def image_dataset(dump_mock_img_data_dir):
    """Initialize the dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return ImageFolderDataset(str(dump_mock_img_data_dir), transform=transform)


@pytest.fixture
def basic_params(root_directory, create_and_save_dummy_imgs):
    # create train, val, and test subdirs
    create_and_save_dummy_imgs(dir_path=osp.join(root_directory, "train"))
    create_and_save_dummy_imgs(dir_path=osp.join(root_directory, "val"))
    create_and_save_dummy_imgs(dir_path=osp.join(root_directory, "test"))
    return {
        "train_transform": lambda x: x,  # Dummy transform
        "val_transform": lambda x: x,
        "test_transform": lambda x: x,
        "root": root_directory,
        "train_path": "train",
        "val_path": "val",
        "test_path": "test"
    }

# ######### Fixtures for ClassifierDataset #########

@pytest.fixture
def mock_classifierdataset(basic_params):
    return ClassifierDataset(data_mode="imgs", **basic_params)

# ######### Fixtures for MnistDataset #########

@pytest.fixture
def mnist_dir():
    return "data"


@pytest.fixture
def sample_batch():
    return torch.rand(10, 1, 28, 28)  # 10 images, 1 channel, 28x28 pixels
