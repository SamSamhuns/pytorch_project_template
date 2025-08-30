from collections.abc import Callable

import pytest
import torch
from omegaconf import OmegaConf

from src.config_parser import CustomDictConfig
from src.trainers import BaseTrainer, ClassifierTrainer
from tests.conftest import NUM_CLS, NUM_IMGS_P_CLS, PYTEST_TEMP_ROOT

CUDA_AVAI = torch.cuda.is_available()


class PatchedBaseTrainer(BaseTrainer):
    """Patches the abstract methods of the BaseTrainer class
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()

    def validate(self):
        raise NotImplementedError()


@pytest.fixture(scope="function", params=[None, pytest.param([0], marks=pytest.mark.skipif(not CUDA_AVAI, reason="CUDA unavailable"))])
def mock_clsf_config(
        request, root_directory, create_and_save_dummy_imgs: Callable) -> CustomDictConfig:
    """Configure and create a training environment for a
    CNN classifier using a custom image dataset
    """
    create_and_save_dummy_imgs(
        root_directory, n_cls=NUM_CLS, n_imgs_p_cls=NUM_IMGS_P_CLS)
    gpu_device = request.param
    cfg_dict = {
        "experiment_name": "pytest_base_trainer",
        "save_dir": f"{PYTEST_TEMP_ROOT}/base_trainer",
        "git_hash": None,
        "mode": "PYTEST",
        "seed": 42,
        "reproducible": True,
        "device": "cuda" if gpu_device else "cpu",
        "gpu_device": gpu_device,
        "use_amp": False,
        "torch_compile_model": False,
        "trainer": {
            "type": "ClassifierTrainer",
            "resume_checkpoint": None,
            "save_best_only": True,
            "batch_log_freq": 2,
            "weight_save_freq": 2,
            "valid_freq": 2,
            "epochs": 4,
            "use_tensorboard": False,
            "tensorboard_port": 6006
        },
        "model": {
            "type": "ClassifierModel",
            "args": {
                "backbone": "mobilenet_v2",
                "feat_extract": False,
                "num_classes": NUM_CLS,
                "pretrained_weights": "MobileNet_V2_Weights.IMAGENET1K_V1"
            },
            "info": {
                "input_width": 100,
                "input_height": 100,
                "input_channel": 3
            }
        },
        "dataset": {
            "type": "ClassifierDataset",
            "args": {
                "root": root_directory,
                "train_path": "train",
                "val_path": "val",
                "test_path": "test"
            },
            "num_classes": NUM_CLS,
            "preprocess": {
                "train_transform": "ImagenetClassifierPreprocess",
                "val_transform": "ImagenetClassifierPreprocess",
                "test_transform": "ImagenetClassifierPreprocess",
                "inference_transform": "ImagenetClassifierPreprocess"
            }
        },
        "dataloader": {
            "type": "CustomDataLoader",
            "args": {
                "batch_size": 16,
                "shuffle": False,
                "num_workers": 4,
                "validation_split": 0.0
            }
        },
        "optimizer": {
            "type": "SGD",
            "args": {"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-05}
        },
        "lr_scheduler": {
            "type": "ReduceLROnPlateau",
            "args": {"factor": 0.1, "patience": 5}
        },
        "loss": {
            "type": "NLLLoss",
            "args": {}
        },
        "metrics": {
            "val": [
                "accuracy_score"
            ],
            "test": [
                "accuracy_score",
                "f1_score",
                "roc_auc_score",
                "confusion_matrix",
                "classification_report"
            ]
        }
    }
    cfg = OmegaConf.create(cfg_dict)
    return CustomDictConfig(cfg)


@pytest.fixture
def mock_logger(mocker):
    return mocker.Mock()


@pytest.fixture()
def base_trainer_and_logger(
        dump_mock_img_data_dir,
        mock_clsf_config: CustomDictConfig,
        mocker):
    """Get a abstract method patched BaseTrainer
    and the associated logger
    """
    mocked_logger = mocker.patch('logging.getLogger')
    trainer = PatchedBaseTrainer(
        config=mock_clsf_config,
        logger_name="base_trainer_logger")
    return trainer, mocked_logger


@pytest.fixture()
def clsf_trainer_and_logger(
        dump_mock_img_data_dir,
        mock_clsf_config: CustomDictConfig,
        mocker):
    """Get a abstract method patched TimeSeriesClassifierTrainer
    and the associated logger
    """
    mocked_logger = mocker.patch('logging.getLogger')
    trainer = ClassifierTrainer(
        config=mock_clsf_config,
        logger_name="time_series_clsf_trainer_logger")
    return trainer, mocked_logger
