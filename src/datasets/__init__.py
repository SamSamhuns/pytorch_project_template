from .classifier_dataset import ClassifierDataset
from .mnist_dataset import MnistDataset

# the dataset.type values a config can name
DATASETS = {"ClassifierDataset": ClassifierDataset, "MnistDataset": MnistDataset}
