import math

import pytest
import torch
from torch.optim import SGD

from src.schedulers import WarmupCosineAnnealingScheduler


def test_lr_warmup():
    model = torch.nn.Linear(10, 2)
    # Initial LR doesn't matter as LambdaLR overrides it.
    optimizer = SGD(model.parameters(), lr=1)
    total_epochs = 30
    warmup_epochs = 5
    scheduler = WarmupCosineAnnealingScheduler(
        optimizer, total_epochs, warmup_epochs)

    # Testing warmup
    expected_warmup_lrs = [
        i / warmup_epochs for i in range(1, warmup_epochs + 1)]
    for epoch in range(warmup_epochs):
        optimizer.step()
        scheduler.step()
        actual_lr = optimizer.param_groups[0]['lr']
        assert pytest.approx(
            actual_lr, 0.01) == expected_warmup_lrs[epoch], f"LR mismatch at warmup epoch {epoch}"


def test_cosine_annealing():
    model = torch.nn.Linear(10, 2)
    optimizer = SGD(model.parameters(), lr=1)
    total_epochs = 100
    warmup_epochs = 5
    scheduler = WarmupCosineAnnealingScheduler(
        optimizer, total_epochs, warmup_epochs)

    # Skip warmup
    for _ in range(warmup_epochs):
        optimizer.step()
        scheduler.step()

    # Testing cosine annealing post-warmup
    for epoch in range(warmup_epochs + 1, total_epochs):
        optimizer.step()
        scheduler.step()
        actual_lr = optimizer.param_groups[0]['lr']
        expected_lr = 0.5 * \
            (1 + math.cos((epoch - warmup_epochs) /
             (total_epochs - warmup_epochs) * math.pi))
        assert pytest.approx(
            actual_lr, 0.01) == expected_lr, f"LR mismatch at epoch {epoch}"


def test_lr_ends_correctly():
    model = torch.nn.Linear(10, 2)
    optimizer = SGD(model.parameters(), lr=1)
    total_epochs = 30
    warmup_epochs = 5
    scheduler = WarmupCosineAnnealingScheduler(
        optimizer, total_epochs, warmup_epochs)

    # Testing the last epoch LR
    for _ in range(total_epochs):
        optimizer.step()
        scheduler.step()

    actual_final_lr = optimizer.param_groups[0]['lr']
    assert pytest.approx(
        actual_final_lr, 0.0001) == 0, "Final LR should be close to 0"
