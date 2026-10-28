from .classifier_model import torchvision_classifier
from .custom_model import CustomModel
from .mnist_model import MnistModel

# the model.type values a config can name
MODELS = {"torchvision_classifier": torchvision_classifier, "MnistModel": MnistModel, "CustomModel": CustomModel}
