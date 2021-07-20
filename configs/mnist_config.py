# imported objects should not be instanitated here
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
    "CUDNN_DETERMINISTIC": True,
    "CUDNN_BENCHMARK": False,
    "GPU_DEVICE": [0],
    "ARCH": {
        "TYPE": Mnist,
        "ARGS": {
            "backbone": None,
            "feat_extract": False,
            "pretrained": False,
        },
        "INPUT_WIDTH": 28,
        "INPUT_HEIGHT": 28,
        "INPUT_CHANNEL": 1
    },
    "DATASET": {
        "TYPE": MnistDataset,
        "NUM_CLASSES": 10,
        "DATA_DIR": {"data_root": "data",
                     "train_dir": "train",
                     "val_dir": None,
                     "test_dir": "test",
                     },
        "PREPROCESS": {"train_transform": Preprocess.train,
                       "val_transform":  Preprocess.val,
                       "test_transform": Preprocess.test,
                       "inference_transform": Preprocess.inference,
                       },
    },
    "DATALOADER": {
        "TYPE": BaseDataLoader,
        "ARGS": {"batch_size": 32,
                 "shuffle": True,
                 "num_workers": 0,
                 "validation_split": 0.1,
                 "pin_memory": True,
                 "drop_last": False,
                 "prefetch_factor": 2,
                 "worker_init_fn": None},
    },
    "OPTIMIZER": {
        "TYPE": optim.SGD,
        "ARGS": {"lr": 1e-2,
                 "momentum": 0.5}
    },
    "LOSS": nn.NLLLoss,
    "METRICS": ["val_accuracy"],
    "LR_SCHEDULER": {
        "TYPE": optim.lr_scheduler.ReduceLROnPlateau,
        "ARGS": {"factor": 0.1,
                 "patience": 8}
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
