"""
An example for the model class
"""
import math
import torch.nn as nn
from .base_model import BaseModel
from src.utils.weights_initializer import weights_init


class CustomModel(BaseModel):
    """
    Example NN model
    """

    def __init__(self, in_c: int, num_classes: int, input_image_size: int, **kwargs):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=32,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by 2x

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by 2x

            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by 2x
        )

        # Compute the size of the flattened feature map
        self.flattened_size = self._compute_flattened_size(input_image_size)

        # Fully connected classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten the feature maps
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )

        # Initialize weights
        self.apply(weights_init)

    def _compute_flattened_size(self, input_size: int) -> int:
        """
        Compute the size of the feature maps after the feature extractor
        given the input image size.
        """
        def conv_output_size(size, kernel_size=3, stride=1, padding=1):
            return math.floor((size + 2 * padding - kernel_size) / stride + 1)

        def pool_output_size(size, kernel_size=2, stride=2):
            return math.floor((size - kernel_size) / stride + 1)

        # Apply the sequence of layers to compute the final size
        size = input_size
        size = conv_output_size(size)  # Conv1
        size = pool_output_size(size)  # MaxPool1
        size = conv_output_size(size)  # Conv2
        size = pool_output_size(size)  # MaxPool2
        size = conv_output_size(size)  # Conv3
        size = pool_output_size(size)  # MaxPool3

        # The final size after feature extraction
        return size * size * 128  # Multiply by the number of channels in the last layer

    def forward(self, x):
        x = self.feature_extractor(x)
        out = self.classifier(x)
        return out
