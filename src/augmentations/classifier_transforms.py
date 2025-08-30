"""Custom model transforms
The class must have a train, val, and test transforms.Compose member
"""
from torchvision import transforms


class ImagenetClassifierPreprocess:
    common_transform = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )]
    train = transforms.Compose(common_transform.copy())
    val = transforms.Compose(common_transform.copy())
    test = transforms.Compose(common_transform.copy())
    inference = transforms.Compose(common_transform.copy())

    def __init__(self):
        """Class to store the train, test, inference transforms or augmentations
        """


class CustomImageClassifierPreprocess:
    """The Normalize transforms is added inside the dataset class for this preprocessor
    after calculating the train set mean and std dev
    """

    common_transform = [
        transforms.Resize(224),
        transforms.ToTensor()]
    train = transforms.Compose(common_transform.copy())
    val = transforms.Compose(common_transform.copy())
    test = transforms.Compose(common_transform.copy())
    inference = transforms.Compose(common_transform.copy())

    def __init__(self):
        """Class to store the train, test, inference transforms or augmentations
        """
