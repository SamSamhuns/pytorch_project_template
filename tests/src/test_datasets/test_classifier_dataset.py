import os
import os.path as osp

import pytest
import torch

from src.datasets import ClassifierDataset
from tests.conftest import PYTEST_TEMP_ROOT


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


def test_plot_samples_per_epoch(mock_classifierdataset):
    """Test if plotting function saves images correctly."""
    fake_batch = torch.rand(16, 3, 224, 224)  # Fake data batch
    out_dir = PYTEST_TEMP_ROOT + "/"
    img_path = f"{out_dir}samples_epoch_{0:d}.png"
    mock_classifierdataset.plot_samples_per_epoch(
        fake_batch, 0, out_dir)
    assert os.path.exists(
        img_path), "Plot image should be saved to the disk."
