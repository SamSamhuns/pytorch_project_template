from torchvision import transforms


class MnistPreprocess:
    common = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    train = transforms.Compose(common.copy())
    val = transforms.Compose(common.copy())
    test = transforms.Compose(common.copy())
    inference = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((28, 28)),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    def __init__(self):
        """Class to store the train, test, inference transforms or augmentations
        """
