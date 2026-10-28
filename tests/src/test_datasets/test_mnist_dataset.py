from torchvision.transforms import ToTensor

from src.datasets import MnistDataset


def test_init_with_validation_split(mnist_dir):
    dataset = MnistDataset(train_transform=ToTensor(), eval_transform=ToTensor(), root=mnist_dir, validation_split=0.1)
    assert len(dataset.train_set) == 54000
    assert len(dataset.val_set) == 6000
    assert len(dataset.test_set) == 10000
