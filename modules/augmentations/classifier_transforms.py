from torchvision import transforms


class ClassifierPreprocess:
    common_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    train = common_transform
    val = common_transform
    test = common_transform
    inference = common_transform

    def __init__(self):
        """
        Class to store the train, test, inference transforms or augmentations
        """
