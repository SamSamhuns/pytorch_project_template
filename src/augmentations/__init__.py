from .numeric import NumerizeLabels
from .classifier_transforms import ImagenetClassifierPreprocess, CustomImageClassifierPreprocess
from .mnist_transforms import MnistPreprocess


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
