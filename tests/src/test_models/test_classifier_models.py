import pytest
import torch

from src.models import ClassifierModel


def test_initialization(classifier_model_params):
    """Test initialization of the ClassifierModel."""
    model = ClassifierModel(**classifier_model_params)
    assert model.model is not None, "Model should be initialized."


def test_feat_extract_mode(classifier_model_params):
    """Test feature extraction mode returns the backbone features instead of class scores."""
    classifier_model_params['feat_extract'] = True
    # No classes when feature extraction is enabled
    classifier_model_params['num_classes'] = 0

    model = ClassifierModel(**classifier_model_params).eval()
    output = model(torch.rand(1, 3, 224, 224))
    assert output.shape == (1, 1280), "Output should be mobilenet_v2's 1280 features."


@pytest.mark.parametrize("backbone", ["resnet18", "mobilenet_v3_small", "convnext_tiny"])
def test_head_replacement_across_backbones(backbone):
    """Only the last Linear layer is replaced, so heads with several layers keep working."""
    model = ClassifierModel(backbone=backbone, num_classes=3, pretrained_weights=None).eval()
    output = model(torch.rand(2, 3, 64, 64))
    assert output.shape == (2, 3)


def test_classifier_mode_output_shape(classifier_model_params, device):
    """Test that the classifier mode produces the correct output shape."""
    model = ClassifierModel(**classifier_model_params)
    model = model.to(device)
    # Single image with the size expected by the backbone
    test_input = torch.rand(1, 3, 224, 224)
    output = model(test_input.to(device))
    assert output.shape == (
        1, 10), "Output shape should match the number of classes."


def test_raises_with_incompatible_parameters():
    """Test that the model raises an error if feature extraction is requested with num_classes set."""
    with pytest.raises(RuntimeError):
        ClassifierModel(feat_extract=True, num_classes=10)
