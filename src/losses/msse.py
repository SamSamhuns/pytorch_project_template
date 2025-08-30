
import torch
import torch.nn as nn


class MSSELoss(nn.Module):
    """Mean of sum of squared errors loss.
    MSSELoss scales the loss unlike MSELoss
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["mean", "sum", None]:
            raise ValueError(
                f"{reduction} is not a valid reduction mode." +
                " Supported reduction modes are'mean','sum' and None")
        if reduction == "mean":
            self.reduction = torch.mean
        elif reduction == "sum":
            self.reduction = torch.sum
        else:
            self.reduction = lambda x: x

    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.sum((outputs - inputs) ** 2,
                           dim=tuple(range(1, outputs.dim())))
        loss = self.reduction(scores)
        return scores, loss
