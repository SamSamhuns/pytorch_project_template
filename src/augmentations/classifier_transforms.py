"""Transforms for torchvision classifiers
Training uses random crops and flips. Evaluation (val, test, inference) uses the resize, crop and
normalization that the pretrained weights expect, e.g. ResNet50_Weights.IMAGENET1K_V2 resizes to 232, not 256
"""

from torchvision import models, transforms


class ImagenetClassifierPreprocess:
    def __init__(self, weights: str | None = None):
        """weights: torchvision weights name, e.g. "MobileNet_V2_Weights.IMAGENET1K_V1".
        None, for a backbone trained from scratch, uses the standard ImageNet eval transform
        """
        # MobileNet_V2_Weights.IMAGENET1K_V1 uses the standard one: resize 256, crop 224, ImageNet mean and std
        self.eval_transform = models.get_weight(weights or "MobileNet_V2_Weights.IMAGENET1K_V1").transforms()
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.eval_transform.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.eval_transform.mean, self.eval_transform.std),
            ]
        )
        # what eval_transform does; export.py stores it in the ONNX file, for the server, which has no torchvision
        self.eval_settings = {
            "image_mode": "RGB",
            "resize_size": self.eval_transform.resize_size[0],
            "crop_size": self.eval_transform.crop_size[0],
            "mean": self.eval_transform.mean,
            "std": self.eval_transform.std,
            "interpolation": self.eval_transform.interpolation.value,  # e.g. bilinear or bicubic
        }
