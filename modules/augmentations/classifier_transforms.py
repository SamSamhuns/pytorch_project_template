from torchvision import transforms


class Preprocess:
    common_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    train = common_transform
    train = common_transform
    inference = common_transform

    def __init__(self):
        """
        Class to store the train, test, inference transforms or augmentations
        """
        pass
