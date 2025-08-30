from .classifier_transforms import CustomImageClassifierPreprocess, ImagenetClassifierPreprocess
from .mnist_transforms import MnistPreprocess
from .numeric import NumerizeLabels

IMPLEMENTED_TRANSFORMS = {
    "NumerizeLabels": NumerizeLabels,
    "ImagenetClassifierPreprocess": ImagenetClassifierPreprocess,
    "CustomImageClassifierPreprocess": CustomImageClassifierPreprocess,
    "MnistPreprocess": MnistPreprocess
}


def init_transform(transform_name: str, **kwargs):
    """Initialize the trainer."""
    try:
        transform = IMPLEMENTED_TRANSFORMS[transform_name](**kwargs)
    except KeyError as exc:
        raise NotImplementedError(
            f"{transform_name} is not implemented. " +
            f"Available Transforms: {IMPLEMENTED_TRANSFORMS.keys()}") from exc
    return transform
