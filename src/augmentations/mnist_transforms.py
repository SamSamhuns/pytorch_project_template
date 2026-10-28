from torchvision import transforms


class MnistPreprocess:
    # Grayscale and Resize do nothing to MNIST digits, but let inference take any RGB image
    common = (
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    )
    train = transforms.Compose(list(common))
    val = transforms.Compose(list(common))
    test = transforms.Compose(list(common))
    inference = transforms.Compose(list(common))

    def __init__(self):
        """Class to store the train, test, inference transforms or augmentations"""
