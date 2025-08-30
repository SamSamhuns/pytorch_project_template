from unittest.mock import patch

import pytest
from torchvision.transforms import ToTensor

from src.datasets import MnistDataset


def test_init_download_mode(mnist_dir):
    dataset = MnistDataset(train_transform=ToTensor(), test_transform=ToTensor(), root=mnist_dir, data_mode="download")
    assert dataset.train_set is not None
    assert dataset.test_set is not None


def test_init_imgs_mode(mnist_dir):
    with pytest.raises(NotImplementedError):
        MnistDataset(root=mnist_dir, data_mode="imgs")


def test_init_numpy_mode(mnist_dir):
    with pytest.raises(NotImplementedError):
        MnistDataset(root=mnist_dir, data_mode="numpy")


def test_init_invalid_mode(mnist_dir):
    with pytest.raises(Exception):
        MnistDataset(root=mnist_dir, data_mode="invalid")


@patch('imageio.imread')
@patch('torchvision.utils.save_image')
def test_plot_samples_per_epoch(mock_save_image, mock_read_image, mnist_dir, sample_batch):
    mock_read_image.return_value = "Image loaded"
    dataset = MnistDataset(root=mnist_dir, data_mode="download")
    result = dataset.plot_samples_per_epoch(sample_batch, 1, mnist_dir)
    mock_save_image.assert_called_once()
    assert result == "Image loaded"


@patch('imageio.mimsave')
@patch('imageio.imread')
def test_make_gif(mock_read_image, mock_mimsave, mnist_dir):
    mock_read_image.side_effect = ["Image 1", "Image 2", "Image 3"]
    dataset = MnistDataset(root=mnist_dir, data_mode="download")
    dataset.make_gif(2, mnist_dir)
    assert mock_read_image.call_count == 3
    mock_mimsave.assert_called_once()
