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
    "USE_CUDA": False,
    "CUDNN_DETERMINISTIC": True,
    "CUDNN_BENCHMARK": False,
    "N_GPU": 1,
    "GPU_DEVICE": [0],
    "ARCH": {
        "TYPE": Classifier,
        "BACKBONE": mobilenet_v2,
        "FEAT_EXTRACT": False,
        "PRETRAINED": False,
        "INPUT_WIDTH": 224,
        "INPUT_HEIGHT": 224,
        "INPUT_CHANNEL": 3
    },
    "DATASET": {
        "TYPE": ClassifierDataset,
        "DATA_ROOT_DIR": "data/birds_dataset",
        "TRAIN_DIR": "train",
        "VAL_DIR": "valid",
        "TEST_DIR": "test",
        "NUM_CLASSES": 265
    },
    "DATALOADER": {
        "TYPE": BaseDataLoader,
        "BATCH_SIZE": 32,
        "SHUFFLE": True,
        "NUM_WORKERS": 0,
        "VALIDATION_SPLIT": 0,
        "PIN_MEMORY": True,
        "PREPROCESS_TRAIN": Preprocess.train,
        "PREPROCESS_VAL": Preprocess.val,
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
