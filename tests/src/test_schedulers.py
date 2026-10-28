import pytest
import torch
from omegaconf import OmegaConf
from torch.optim import SGD

from src.trainers import build_scheduler


def lrs_per_epoch(scheduler_cfg: dict, epochs: int) -> list[float]:
    """The LR each epoch trains with, for a base LR of 1"""
    optimizer = SGD(torch.nn.Linear(2, 2).parameters(), lr=1.0)
    scheduler = build_scheduler(optimizer, OmegaConf.create(scheduler_cfg))
    lrs = []
    for _ in range(epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return lrs


def test_cosine_annealing():
    lrs = lrs_per_epoch({"type": "CosineAnnealingLR", "args": {"T_max": 4}}, epochs=5)
    assert lrs == pytest.approx([1.0, 0.854, 0.5, 0.146, 0.0], abs=1e-3)


def test_warmup_then_cosine():
    """The first epoch trains at 10% of the LR, not at 0, and the cosine starts after the warmup"""
    lrs = lrs_per_epoch({"type": "CosineAnnealingLR", "args": {"T_max": 2}, "warmup_epochs": 2}, epochs=5)
    assert lrs == pytest.approx([0.1, 0.55, 1.0, 0.5, 0.0], abs=1e-3)
