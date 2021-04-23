from torchvision import transforms


class Preprocess:

    __slots__ = ['train', 'test', 'inference']

    def __init__(self):
        """
        Class to store the train, test, inference transforms or augmentations
        """
        common_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
        self.train = common_transform
        self.train = common_transform
        self.inference = common_transform
