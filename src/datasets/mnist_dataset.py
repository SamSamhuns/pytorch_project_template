"""Mnist Data loader"""

from torch.utils.data import random_split
from torchvision import datasets


class MnistDataset:
    def __init__(self, train_transform=None, eval_transform=None, root="data", validation_split=0.0, generator=None):
        """train_transform: torchvision.transforms for train data
        eval_transform: torchvision.transforms for validation and test data
        root: folder MNIST is downloaded to
        validation_split: fraction of the train data to validate on, as MNIST has no val set
        generator: torch.Generator that seeds the validation split
        """
        self.train_set = datasets.MNIST(root, train=True, download=True, transform=train_transform)
        self.val_set = None
        if validation_split > 0:
            self.train_set, self.val_set = random_split(
                self.train_set, [1 - validation_split, validation_split], generator=generator
            )
            # validate on the same images loaded with the eval transform, so they aren't augmented
            self.val_set.dataset = datasets.MNIST(root, train=True, transform=eval_transform)
        self.test_set = datasets.MNIST(root, train=False, transform=eval_transform)
