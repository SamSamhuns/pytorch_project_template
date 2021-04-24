from torchvision import transforms


class Preprocess:
    train = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
    val = train
    test = train
    inference = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((28, 28)),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    def __init__(self):
        """
        Class to store the train, test, inference transforms or augmentations
        """
        pass
