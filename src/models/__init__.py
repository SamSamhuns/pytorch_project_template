from .classifier_model import torchvision_classifier
from .custom_model import CustomModel
from .mnist_model import MnistModel

IMPLEMENTED_MODELS = {
    "torchvision_classifier": torchvision_classifier,
    "MnistModel": MnistModel,
    "CustomModel": CustomModel,
}


def init_model(model_name: str, **kwargs):
    """Initialize the model."""
    try:
        model = IMPLEMENTED_MODELS[model_name](**kwargs)
    except KeyError as exc:
        raise NotImplementedError(
            f"{model_name} is not implemented. " + f"Available Models: {IMPLEMENTED_MODELS.keys()}"
        ) from exc
    return model
