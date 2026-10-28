"""Mnist Data loader"""

from torchvision import datasets


class MnistDataset:
    def __init__(self, train_transform=None, test_transform=None, root="data", data_mode="download", **kwargs):
        """train_transform: torchvision.transforms for train data
        test_transform: torchvision.transforms for test data
        root: folder containing train & test data dirs
        data_mode: Mode for getting data, only "download" is supported
        """
        if data_mode != "download":
            raise NotImplementedError(f"{data_mode} data_mode is not supported. Available modes are: download")
        self.val_set = None
        self.train_set = datasets.MNIST(root, train=True, download=True, transform=train_transform)
        self.test_set = datasets.MNIST(root, train=False, transform=test_transform)
