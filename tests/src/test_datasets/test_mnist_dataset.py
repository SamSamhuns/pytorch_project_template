from torchvision.datasets import FakeData

from src.datasets import MnistDataset


class FakeMNIST(FakeData):
    """Random images in place of MNIST, so the test needs no download"""

    def __init__(self, root, train, transform=None, download=False):
        super().__init__(size=600 if train else 100, image_size=(1, 28, 28), num_classes=10, transform=transform)
        self.classes = [f"{digit} - digit" for digit in range(10)]


def test_init_with_validation_split(monkeypatch):
    """validation_split carves the val set out of the train set, and the val set gets the eval transform"""
    monkeypatch.setattr("src.datasets.mnist_dataset.datasets.MNIST", FakeMNIST)
    dataset = MnistDataset(train_transform=lambda img: "train", eval_transform=lambda img: "eval", validation_split=0.1)

    assert (len(dataset.train_set), len(dataset.val_set), len(dataset.test_set)) == (540, 60, 100)
    assert dataset.train_set[0][0] == "train"
    assert dataset.val_set[0][0] == dataset.test_set[0][0] == "eval"
    assert dataset.classes[7] == "7 - digit"
