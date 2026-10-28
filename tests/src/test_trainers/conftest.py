import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from src.trainers import ClassifierTrainer
from tests.conftest import NUM_CLS

CLASSES = [f"class_{i}" for i in range(NUM_CLS)]


def tiny_model(num_classes: int = NUM_CLS) -> nn.Module:
    """A model small enough to train in a test"""
    return nn.Sequential(nn.Conv2d(3, 4, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, num_classes))


@pytest.fixture
def trainer_cfg():
    """The parts of a config that the trainer reads"""
    return OmegaConf.create(
        {
            "use_amp": False,
            "torch_compile_model": False,
            "trainer": {"batch_log_freq": 2, "valid_freq": 1, "epochs": 2, "use_tensorboard": False},
            "optimizer": {"type": "SGD", "args": {"lr": 0.01, "momentum": 0.9}},
            "loss": {"type": "CrossEntropyLoss", "args": {}},
            "lr_scheduler": {"type": "CosineAnnealingLR", "args": {"T_max": 2}},
            "metrics": {
                "average": "macro",
                "val": ["accuracy_score"],
                "test": ["accuracy_score", "f1_score", "confusion_matrix", "classification_report"],
            },
        }
    )


@pytest.fixture
def run_dir(tmp_path) -> str:
    """A run dir with the models and logs dirs the trainer writes to"""
    (tmp_path / "models").mkdir()
    (tmp_path / "logs").mkdir()
    return str(tmp_path)


@pytest.fixture
def loader() -> DataLoader:
    """32 random 16x16 images of NUM_CLS classes"""
    return DataLoader(FakeData(32, (3, 16, 16), num_classes=NUM_CLS, transform=ToTensor()), batch_size=8)


@pytest.fixture(
    params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))]
)
def device(request) -> torch.device:
    return torch.device(request.param)


@pytest.fixture
def trainer(trainer_cfg, run_dir) -> ClassifierTrainer:
    return ClassifierTrainer(trainer_cfg, tiny_model(), run_dir, torch.device("cpu"), classes=CLASSES)
