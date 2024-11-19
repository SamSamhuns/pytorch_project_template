import torch
import pytest

from src.models import BaseModel, CustomModel, MnistModel


@pytest.fixture(params=[
    torch.device('cpu'),
    pytest.param(torch.device('cuda'), marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))])
def device(request):
    return request.param


@pytest.fixture(params=[torch.float32])
def dtype(request):
    return request.param


@pytest.fixture
def base_model():
    """Instance of BaseModel."""
    return BaseModel()


@pytest.fixture
def custom_model():
    """Instance of CustomModel with the provided configuration."""
    return CustomModel(3, 10, 64)


@pytest.fixture
def classifier_model_params():
    """Common parameters for creating a ClassifierModel."""
    return {
        "backbone": "mobilenet_v2",
        "num_classes": 10,
        "feat_extract": False,
        "pretrained_weights": "MobileNet_V2_Weights.IMAGENET1K_V1"
    }


@pytest.fixture
def mnist_model():
    """Create an instance of MnistModel for testing."""
    return MnistModel()
