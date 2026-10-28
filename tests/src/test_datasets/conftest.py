import os.path as osp

import pytest
from torchvision import transforms

from src.datasets import ClassifierDataset

# ######### Fixtures for ClassifierDataset #########


@pytest.fixture
def basic_params(root_directory, create_and_save_dummy_imgs):
    # create train, val, and test subdirs
    create_and_save_dummy_imgs(dir_path=osp.join(root_directory, "train"))
    create_and_save_dummy_imgs(dir_path=osp.join(root_directory, "val"))
    create_and_save_dummy_imgs(dir_path=osp.join(root_directory, "test"))
    return {
        "train_transform": transforms.Compose([transforms.ToTensor()]),  # Dummy transform
        "eval_transform": transforms.Compose([transforms.ToTensor()]),
        "root": root_directory,
        "train_path": "train",
        "val_path": "val",
        "test_path": "test",
    }


@pytest.fixture
def mock_classifierdataset(basic_params):
    return ClassifierDataset(data_mode="imgs", **basic_params)
