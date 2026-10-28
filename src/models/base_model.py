
from torch import nn
from torchinfo import summary


class BaseModel(nn.Module):
    """Base class for all neural networks."""

    def __init__(self):
        super().__init__()

    def forward(self, *input_x):
        """Forward pass logic
        :return: Network output
        """
        raise NotImplementedError

    def print_summary(self, input_size: tuple[int], device: str = "cpu"):
        """Generate Network summary.
        input_size: tuple of input size [bsize, ...]
        device: str should be either 'cuda' or 'cpu'
        """
        assert device in {"cuda", "cpu"}
        summary(self, input_size=input_size, device=device)
