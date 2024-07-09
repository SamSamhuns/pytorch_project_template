from torch.nn import NLLLoss, CrossEntropyLoss, MSELoss, BCELoss, BCEWithLogitsLoss
from .msse import MSSELoss


IMPLEMENTED_LOSSES = {"NLLLoss": NLLLoss,
                      "CrossEntropyLoss": CrossEntropyLoss,
                      "BCELoss": BCELoss,
                      "BCEWithLogitsLoss": BCEWithLogitsLoss,
                      "MSELoss": MSELoss,
                      "MSSELoss": MSSELoss}


def init_loss(loss_name: str, **kwargs):
    """Initialize the loss."""
    try:
        loss = IMPLEMENTED_LOSSES[loss_name](**kwargs)
    except KeyError as exc:
        raise NotImplementedError(
            f"{loss_name} is not implemented. " +
            f"Available Losses: {IMPLEMENTED_LOSSES.keys()}") from exc
    return loss
