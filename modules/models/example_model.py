"""
An example for the model class
"""
import torch.nn as nn
from .base_model import BaseModel
from modules.utils.weights_initializer import weights_init


class Example(BaseModel):
    """
    Example NN model
    """
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        # inputs are destroyed without allocating additional output
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=self.config.INPUT_CHANNELs,
                              out_channels=self.config.num_filters,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)

        # initialize weights
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        out = x.view(x.size(0), -1)
        return out
