"""Transforms for torchvision classifiers
Training uses random crops and flips. Evaluation (val, test, inference) uses the resize, crop and
normalization that the pretrained weights expect, e.g. ResNet50_Weights.IMAGENET1K_V2 resizes to 232, not 256
"""

from collections.abc import Callable

from torchvision import models, transforms


def imagenet_transforms(weights: str | None = None) -> tuple[Callable, Callable, dict]:
    """The train transform, the eval transform and what the eval transform does, its settings.
    weights: torchvision weights name, e.g. "MobileNet_V2_Weights.IMAGENET1K_V1".
    None, for a backbone trained from scratch, uses the standard ImageNet eval transform
    """
    # MobileNet_V2_Weights.IMAGENET1K_V1 uses the standard one: resize 256, crop 224, ImageNet mean and std
    eval_transform = models.get_weight(weights or "MobileNet_V2_Weights.IMAGENET1K_V1").transforms()
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(eval_transform.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(eval_transform.mean, eval_transform.std),
        ]
    )
    # export.py stores the settings in the ONNX file, for the server, which has no torchvision
    eval_settings = {
        "image_mode": "RGB",
        "resize_size": eval_transform.resize_size[0],
        "crop_size": eval_transform.crop_size[0],
        "mean": eval_transform.mean,
        "std": eval_transform.std,
        "interpolation": eval_transform.interpolation.value,  # e.g. bilinear or bicubic
    }
    return train_transform, eval_transform, eval_settings
