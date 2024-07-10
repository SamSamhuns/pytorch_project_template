"""
An example for the model class
"""
import torch.nn as nn
from .base_model import BaseModel
from src.utils.weights_initializer import weights_init


class CustomModel(BaseModel):
    """
    Example NN model
    """
    def __init__(self, in_c: int, out_c: int, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)
        self.relu = nn.ReLU(inplace=True)

        # initialize weights
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        out = x.view(x.size(0), -1)
        return out
