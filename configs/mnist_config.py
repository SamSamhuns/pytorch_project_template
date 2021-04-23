from torch import nn
import torch.optim as optim

from modules.models.mnist_model import Mnist
from modules.datasets.mnist_dataset import MnistDataset
from modules.dataloaders.base_dataloader import BaseDataLoader
from modules.augmentations.mnist_transforms import Preprocess


CONFIG = {
    "NAME": "mnist_classifier",
    "SEED": 1,
    "USE_CUDA": False,
    "N_GPU": 1,
    "GPU_DEVICE": [0],
    "ARCH": {
        "TYPE": Mnist,
        "BACKBONE": None,
        "FEAT_EXTRACT": False,
        "PRETRAINED": False,
        "INPUT_WIDTH": 28,
        "INPUT_HEIGHT": 28,
        "INPUT_CHANNEL": 1
    },
    "DATASET": {
        "TYPE": MnistDataset,
        "DATA_ROOT_DIR": "data",
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
        "PIN_MEMORY": True,
        "PREPROCESS_TRAIN": Preprocess.train,
        "PREPROCESS_TEST": Preprocess.test,
        "PREPROCESS_INFERENCE": Preprocess.inference
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
        "RESUME": True,
        "SAVE_BEST_ONLY": True,
        "LOG_FREQ": 50,
        "VALID_FREQ": 2,

        "EPOCHS": 12,
        "CHECKPOINT_DIR": "checkpoints",

        "VERBOSITY": 2,
        "EARLY_STOP": 10,
        "USE_TENSORBOARD": True,
        "TENSORBOARD_EXPERIMENT_DIR": "experiments",
        "TENSORBOARD_PORT": 6006
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
