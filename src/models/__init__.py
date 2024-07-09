from .base_model import BaseModel
from .classifier_model import ClassifierModel
from .mnist_model import MnistModel
from .custom_model import CustomModel


IMPLEMENTED_MODELS = {
    "ClassifierModel": ClassifierModel,
    "MnistModel": MnistModel,
    "CustomModel": CustomModel
}


def init_model(model_name: str, **kwargs):
    """Initialize the model."""
    try:
        model = IMPLEMENTED_MODELS[model_name](**kwargs)
    except KeyError as exc:
        raise NotImplementedError(
            f"{model_name} is not implemented. " +
            f"Available Models: {IMPLEMENTED_MODELS.keys()}") from exc
    return model
