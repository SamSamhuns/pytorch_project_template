import pytest
import torch
from src.losses import MSSELoss


def test_initialization():
    """Test initialization of MSSELoss"""
    # Test valid reductions
    try:
        MSSELoss(reduction='mean')
        MSSELoss(reduction='sum')
        MSSELoss(reduction=None)
    except ValueError:
        pytest.fail("Initialization with valid reduction failed.")

    # Test invalid reduction
    with pytest.raises(ValueError):
        MSSELoss(reduction='invalid')


def test_forward_simple():
    """Test forward pass of MSSELoss"""
    # Create a simple test case
    inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
    targets = torch.tensor([1, 2, 3], dtype=torch.float32)
    loss_module = MSSELoss(reduction='mean')

    scores, loss = loss_module(inputs, targets)
    assert torch.allclose(scores, torch.tensor(
        [0, 0, 0], dtype=torch.float32)), "Scores should be zeros"
    assert torch.allclose(loss, torch.tensor(
        0, dtype=torch.float32)), "Loss should be zero"


def test_reduction_modes():
    inputs = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    targets = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)

    # Test 'sum' reduction
    loss_module_sum = MSSELoss(reduction='sum')
    _, loss_sum = loss_module_sum(inputs, targets)
    expected_loss_sum = torch.sum(
        torch.tensor([0, 1, 4, 9], dtype=torch.float32))
    assert torch.allclose(loss_sum, expected_loss_sum), \
        f"Expected sum reduction loss {expected_loss_sum}, got {loss_sum}"

    # Test 'mean' reduction
    loss_module_mean = MSSELoss(reduction='mean')
    _, loss_mean = loss_module_mean(inputs, targets)
    expected_loss_mean = torch.mean(
        torch.tensor([1, 13], dtype=torch.float32))
    assert torch.allclose(loss_mean, expected_loss_mean), \
        f"Expected mean reduction loss {expected_loss_mean}, got {loss_mean}"

    # Test 'None' reduction (elementwise)
    loss_module_none = MSSELoss(reduction=None)
    scores_none, _ = loss_module_none(inputs, targets)
    expected_scores_none = torch.tensor(
        [1, 13], dtype=torch.float32)
    assert torch.allclose(scores_none, expected_scores_none), \
        "Scores should match elementwise losses"
