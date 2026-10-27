
from torch import nn


class BaseModel(nn.Module):
    """Base class for all neural networks."""

    def __init__(self):
        super().__init__()

    def forward(self, *input_x):
        """Forward pass logic
        :return: Network output
        """
        raise NotImplementedError
