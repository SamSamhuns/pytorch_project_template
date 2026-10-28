"""An example for the model class"""

import torch.nn as nn


class CustomModel(nn.Module):
    """Example CNN. Global average pooling before the head lets it take images of any size"""

    def __init__(self, in_c: int, num_classes: int):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by 2x
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by 2x
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by 2x
        )

        # Fully connected classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # the mean of each of the 128 feature maps
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),  # logits
        )

    def forward(self, x):
        return self.classifier(self.feature_extractor(x))
