"""
Tests for src.utils.statistics
"""
import torch
from src.utils.statistics import get_model_params


def test_get_model_params(simple_1d_conv_model):
    params = get_model_params(simple_1d_conv_model)
    assert params['Total params'] == sum(torch.numel(
        p) for p in simple_1d_conv_model.parameters()), "Total params calculation error"
    assert params['Trainable params'] == sum(torch.numel(
        p) for p in simple_1d_conv_model.parameters(
    ) if p.requires_grad), "Trainable params calculation error"
    assert params['Non-trainable params'] == 0, \
        "There should be no non-trainable params in this model"
