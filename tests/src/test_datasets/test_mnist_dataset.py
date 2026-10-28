import pytest
from torchvision.transforms import ToTensor

from src.datasets import MnistDataset


def test_init_download_mode(mnist_dir):
    dataset = MnistDataset(train_transform=ToTensor(), test_transform=ToTensor(), root=mnist_dir, data_mode="download")
    assert dataset.train_set is not None
    assert dataset.test_set is not None


def test_init_invalid_mode(mnist_dir):
    with pytest.raises(NotImplementedError):
        MnistDataset(root=mnist_dir, data_mode="imgs")
