from torch.nn import CrossEntropyLoss

# the models output logits, which CrossEntropyLoss turns into log-probabilities itself
IMPLEMENTED_LOSSES = {"CrossEntropyLoss": CrossEntropyLoss}


def init_loss(loss_name: str, **kwargs):
    """Initialize the loss."""
    try:
        loss = IMPLEMENTED_LOSSES[loss_name](**kwargs)
    except KeyError as exc:
        raise NotImplementedError(
            f"{loss_name} is not implemented. " + f"Available Losses: {IMPLEMENTED_LOSSES.keys()}"
        ) from exc
    return loss
