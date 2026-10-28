from collections.abc import Callable

from torchvision import transforms


def mnist_transforms() -> tuple[Callable, Callable, dict]:
    """The train transform, the eval transform and what the eval transform does, its settings.
    MNIST trains without augmentation, so both transforms are the same
    """
    # Grayscale, Resize and CenterCrop do nothing to MNIST digits, but let inference take any RGB image
    eval_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    # export.py stores the settings in the ONNX file, for the server, which has no torchvision
    eval_settings = {
        "image_mode": "L",
        "resize_size": 28,
        "crop_size": 28,
        "mean": [0.1307],
        "std": [0.3081],
        "interpolation": "bilinear",
    }
    return eval_transform, eval_transform, eval_settings
