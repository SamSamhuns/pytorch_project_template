from torchvision import transforms


class MnistPreprocess:
    # Grayscale and Resize do nothing to MNIST digits, but let inference take any RGB image
    common = [transforms.Grayscale(),
              transforms.Resize((28, 28)),
              transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,))]
    train = transforms.Compose(common.copy())
    val = transforms.Compose(common.copy())
    test = transforms.Compose(common.copy())
    inference = transforms.Compose(common.copy())

    def __init__(self):
        """Class to store the train, test, inference transforms or augmentations
        """
