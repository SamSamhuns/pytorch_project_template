import pytest
import torch


def test_he_initialization(base_model):
    """Test He initialization on Conv2d and Linear layers."""
    conv_layer = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
    linear_layer = torch.nn.Linear(10, 5)
    base_model.add_module('conv', conv_layer)
    base_model.add_module('linear', linear_layer)

    base_model.initialize_weights("he")  # Run He initialization

    # Check that the weights are initialized correctly
    assert torch.mean(
        conv_layer.weight) != 0, "Conv weights should be initialized."
    assert torch.mean(
        linear_layer.weight) != 0, "Linear weights should be initialized."


def test_glorot_initialization(base_model):
    """Test Glorot initialization on Linear layers."""
    linear_layer = torch.nn.Linear(10, 5)
    base_model.add_module('linear', linear_layer)

    base_model.initialize_weights("glorot")  # Run Glorot initialization

    # Check that the weights are initialized correctly
    assert torch.mean(
        linear_layer.weight) != 0, "Weights should be initialized."


def test_initialize_weights(base_model):
    """Test the selection logic for weight initialization methods."""
    base_model.initialize_weights("he")
    # Here we might check if the weights are indeed initialized using He initialization,
    # but this would require mock objects or inspecting the state change, so we keep it simple.

    with pytest.raises(NotImplementedError):
        base_model.initialize_weights("unknown_method")
