"""Custom model transforms
The class must have a train, val, and test transforms.Compose member
"""

from torchvision import transforms


class ImagenetClassifierPreprocess:
    common_transform = (
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    )
    train = transforms.Compose(list(common_transform))
    val = transforms.Compose(list(common_transform))
    test = transforms.Compose(list(common_transform))
    inference = transforms.Compose(list(common_transform))

    def __init__(self):
        """Class to store the train, test, inference transforms or augmentations"""
