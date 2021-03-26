
from torch import nn
import torch.optim as optim

from modules.models.mnist_model import Mnist
from modules.datasets.mnist_dataset import MnistDataset
from modules.dataloaders.base_dataloader import BaseDataLoader


MNIST_CONFIG = {
    "NAME": "mnist_classifier",
    "SEED": 1,
    "USE_CUDA": False,
    "N_GPU": 1,
    "GPU_DEVICE": [0],
    "ARCH": {
        "TYPE": Mnist,
        "PRETRAINED": False,
        "INPUT_WIDTH": 28,
        "INPUT_HEIGHT": 28
    },
    "DATASET": {
        "TYPE": MnistDataset("download"),
        "RAW_DATA_ROOT_DIR": "data",
        "PROC_DATA_ROOT_DIR": "data",
        "TRAIN_DIR": "train",
        "TEST_DIR": "test",
        "NUM_CLASSES": 10
    },
    "DATALOADER": {
        "TYPE": BaseDataLoader,
        "BATCH_SIZE": 32,
        "SHUFFLE": True,
        "NUM_WORKERS": 0,
        "VALIDATION_SPLIT": 0.1,
        "PIN_MEMORY": True
    },
    "OPTIMIZER": {
        "TYPE": optim.SGD,
        "LR": 1e-2,
        "WEIGHT_DECAY": 0,
        "AMSGRAD": False,
        "MOMENTUM": 0.5
    },
    "LOSS": nn.NLLLoss,
    "METRICS": ["val_accuracy"],
    "LR_SCHEDULER": {
        "TYPE": optim.lr_scheduler.ReduceLROnPlateau,
        "FACTOR": 0.1,
        "PATIENCE": 8
    },
    "TRAINER": {
        "RESUME": False,
        "LOG_FREQ": 50,
        "VALID_FREQ": 2,

        "EPOCHS": 12,
        "CHECKPOINT_DIR": "checkpoints",

        "VERBOSITY": 2,
        "EARLY_STOP": 10,
        "TENSORBOARD": True,
        "TENSORBOARD_PORT": 8099
    },
    "LOGGER": {
        "DIR": "logs",
        "LOG_FMT": "mnist_log_{}.txt",
        "FILE_FMT": "%(asctime)s %(levelname)-8s: %(message)s",
        "CONSOLE_FMT": "%(message)s",

        "<logger levels>": "DEBUG:10, INFO:20, ERROR:40",
        "LOGGER_LEVEL": 10,
        "FILE_LEVEL": 10,
        "CONSOLE_LEVEL": 10
    }
}
