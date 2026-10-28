from torchvision import transforms


class MnistPreprocess:
    def __init__(self):
        # Grayscale, Resize and CenterCrop do nothing to MNIST digits, but let inference take any RGB image
        self.eval_transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(28),
                transforms.CenterCrop(28),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.train_transform = self.eval_transform  # MNIST trains without augmentation
        # what eval_transform does; export.py stores it in the ONNX file, for the server, which has no torchvision
        self.eval_settings = {
            "image_mode": "L",
            "resize_size": 28,
            "crop_size": 28,
            "mean": [0.1307],
            "std": [0.3081],
            "interpolation": "bilinear",
        }
