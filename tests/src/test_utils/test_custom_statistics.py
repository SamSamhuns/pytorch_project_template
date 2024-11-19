"""
Tests for src.utils. custom_statistics
"""
import torch
from src.utils.custom_statistics import get_model_params, get_img_dset_mean_std
from tests.src.test_utils.conftest import DummyDataset


def test_get_model_params(simple_2d_conv_model):
    params = get_model_params(simple_2d_conv_model)
    assert params['Total params'] == sum(torch.numel(
        p) for p in simple_2d_conv_model.parameters()), "Total params calculation error"
    assert params['Trainable params'] == sum(torch.numel(
        p) for p in simple_2d_conv_model.parameters(
    ) if p.requires_grad), "Trainable params calculation error"
    assert params['Non-trainable params'] == 0, \
        "There should be no non-trainable params in this model"


def test_get_img_dset_mean_std():
    # Create a dummy dataset
    samples = 10
    image_size = (32, 32)
    dataset = DummyDataset(samples, image_size)

    # Calculate mean and std using the online method
    online_mean, online_std = get_img_dset_mean_std(dataset, method="online")

    # Calculate mean and std using the offline method
    offline_mean, offline_std = get_img_dset_mean_std(dataset, method="offline")
    # Assert that both methods produce similar results
    assert torch.allclose(online_mean, offline_mean,
                          atol=1e-6), f"Mean mismatch: {online_mean} vs {offline_mean}"
    assert torch.allclose(online_std, offline_std,
                          atol=1e-4), f"Std mismatch: {online_std} vs {offline_std}"
