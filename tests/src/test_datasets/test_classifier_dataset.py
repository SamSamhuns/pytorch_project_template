import os
import os.path as osp
import torch
import pytest

from src.datasets import ClassifierDataset


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


@pytest.fixture
def mock_classifierdataset(basic_params):
    return ClassifierDataset(data_mode="imgs", **basic_params)


def test_initialization_imgs(mock_classifierdataset):
    """Test initialization with image folders."""
    assert mock_classifierdataset.train_set is not None, "Training set should be initialized."
    assert mock_classifierdataset.val_set is not None, "Val set should be initialized."
    assert mock_classifierdataset.test_set is not None, "Test set should be initialized."


def test_initialization_webdataset(basic_params, mock_webdataset_path):
    """Test initialization with webdataset."""
    wdset_path = mock_webdataset_path
    basic_params["train_path"] = osp.basename(wdset_path)
    basic_params["val_path"] = None
    basic_params["test_path"] = None
    dataset = ClassifierDataset(data_mode="webdataset", **basic_params)
    assert dataset.train_set is not None, "Training set should be initialized."


def test_non_implemented_mode(basic_params):
    """Test initialization with a non-implemented mode."""
    with pytest.raises(NotImplementedError):
        _ = ClassifierDataset(data_mode="numpy", **basic_params)


def test_plot_samples_per_epoch(mock_classifierdataset, tmpdir):
    """Test if plotting function saves images correctly."""
    fake_batch = torch.rand(16, 3, 224, 224)  # Fake data batch
    out_dir = str(tmpdir) + "/"
    img_path = f"{out_dir}samples_epoch_{0:d}.png"
    mock_classifierdataset.plot_samples_per_epoch(
        fake_batch, 0, out_dir)
    assert os.path.exists(
        img_path), "Plot image should be saved to the disk."
