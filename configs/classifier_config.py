# imported objects should not be instanitated here
from torch import nn
import torch.optim as optim
from torchvision.models import mobilenet_v2

from modules.models.classifier_model import Classifier
from modules.datasets.classifier_dataset import ClassifierDataset
from modules.dataloaders.base_dataloader import BaseDataLoader
from modules.augmentations.classifier_transforms import Preprocess


CONFIG = {
    "NAME": "image_classifier",
    "SEED": 1,
    "USE_CUDA": False,            # set to True for gpu training
    "CUDNN_DETERMINISTIC": True,  # for repeating results together with SEED
    "CUDNN_BENCHMARK": False,     # set to True for faster training with gpu
    "GPU_DEVICE": [0],            # cuda device list for single/multi gpu training
    "USE_AMP": False,             # automatic mixed precision training for faster train
    "ARCH": {
        "TYPE": Classifier,
        "ARGS": {
            "backbone": mobilenet_v2,
            "feat_extract": False,
            "pretrained": False,
        },
        "INPUT_WIDTH": 224,
        "INPUT_HEIGHT": 224,
        "INPUT_CHANNEL": 3
    },
    "DATASET": {
        "TYPE": ClassifierDataset,
        "NUM_CLASSES": 265,
        "DATA_DIR": {"data_root": "data/birds_dataset",
                     "train_dir": "train",
                     "val_dir": "valid",
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
                 "validation_split": 0.,
                 "pin_memory": True,
                 "drop_last": False,
                 "prefetch_factor": 2,
                 "worker_init_fn": None
                 },
    },
    "OPTIMIZER": {
        "TYPE": optim.SGD,
        "ARGS": {"lr": 1e-2,
                 "momentum": 0.9}
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
        "LOG_FREQ": 5,
        "VALID_FREQ": 2,

        "EPOCHS": 12,
        "CHECKPOINT_DIR": "checkpoints_birds",

        "VERBOSITY": 2,
        "EARLY_STOP": 10,
        "USE_TENSORBOARD": True,
        "TENSORBOARD_EXPERIMENT_DIR": "experiments_birds",
        "TENSORBOARD_PORT": 6006
    },
    "LOGGER": {
        "DIR": "logs_birds",
        "LOG_FMT": "classifier_log_{}.txt",
        "FILE_FMT": "%(asctime)s %(levelname)-8s: %(message)s",
        "CONSOLE_FMT": "%(message)s",

        "<logger levels>": "DEBUG:10, INFO:20, ERROR:40",
        "LOGGER_LEVEL": 10,
        "FILE_LEVEL": 10,
        "CONSOLE_LEVEL": 10
    }
}
