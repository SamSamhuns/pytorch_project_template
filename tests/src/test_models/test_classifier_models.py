import pytest
import torch

from src.models import ClassifierModel



def test_initialization(classifier_model_params):
    """Test initialization of the ClassifierModel."""
    model = ClassifierModel(**classifier_model_params)
    assert model.model is not None, "Model should be initialized."


def test_feat_extract_mode(classifier_model_params):
    """Test feature extraction mode disables the final classifier layer."""
    classifier_model_params['feat_extract'] = True
    # No classes when feature extraction is enabled
    classifier_model_params['num_classes'] = 0

    model = ClassifierModel(**classifier_model_params)
    final_layer = getattr(model.model, 'classifier',
                          None) or getattr(model.model, 'fc', None)
    assert isinstance(
        final_layer, torch.nn.Identity), "Final layer should be nn.Identity in feature extraction mode."


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
