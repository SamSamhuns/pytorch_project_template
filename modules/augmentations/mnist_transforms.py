from torchvision import transforms


class Preprocess:

    __slots__ = ['train', 'test', 'inference']

    def __init__(self):
        """
        Class to store the train, test, inference transforms or augmentations
        """
        self.train = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])
        self.train = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])
        self.inference = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((28, 28)),
                                             transforms.Normalize((0.1307,), (0.3081,))])
