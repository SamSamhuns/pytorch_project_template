from torchvision import transforms


class MnistPreprocess:
    def __init__(self):
        # Grayscale and Resize do nothing to MNIST digits, but let inference take any RGB image
        self.eval_transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.train_transform = self.eval_transform  # MNIST trains without augmentation
