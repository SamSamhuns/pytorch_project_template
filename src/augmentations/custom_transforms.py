"""
Custom model transforms
The class must have a train, val, and test transforms.Compose member
"""

from torchvision import transforms


class CustomImageClassifierPreprocess:
    """
    The Normalize transforms is added inside the dataset class for this preprocessor
    after calculating the train set mean and std dev
    """
    common_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    train = common_transform
    val = common_transform
    test = common_transform
    inference = common_transform

    def __init__(self):
        """
        Class to store the train, test, inference transforms or augmentations
        """
