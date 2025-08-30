
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

    def _he_init(self):
        """Apply He (Kaiming) normal initialization"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                if layer.weight is not None:
                    nn.init.constant_(layer.weight, 1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def _glorot_init(self):
        """Apply Glorot (Xavier) uniform initialization"""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                if layer.weight is not None:
                    nn.init.constant_(layer.weight, 1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def initialize_weights(self, method: str = "he"):
        """Initialize encoder and head weights randomly."""
        if method == "he":
            self._he_init()
        elif method == "glorot":
            self._glorot_init()
        else:
            raise NotImplementedError(f"{method} initialization not implemented")

    def print_summary(self, input_size: tuple[int], device: str = "cpu"):
        """Generate Network summary.
        input_size: tuple of input size [bsize, ...]
        device: str should be either 'cuda' or 'cpu'
        """
        assert device in {"cuda", "cpu"}
        summary(self, input_size=input_size, device=device)
