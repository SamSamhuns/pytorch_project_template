from .base_dataset import BaseDataset
from .mnist_dataset import MnistDataset
from .classifier_dataset import ClassifierDataset


IMPLEMENTED_DATASETS = {
    "MnistDataset": MnistDataset,
    "ClassifierDataset": ClassifierDataset
}


def init_dataset(dataset_name: str, **kwargs):
    """Initialize the dataset."""
    try:
        dataset = IMPLEMENTED_DATASETS[dataset_name](**kwargs)
    except KeyError as exc:
        raise NotImplementedError(
            f"{dataset_name} is not implemented. " +
            f"Available Datasets: {IMPLEMENTED_DATASETS.keys()}") from exc
    return dataset
