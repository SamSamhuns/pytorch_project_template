from torch.nn import CrossEntropyLoss, NLLLoss

# the models output log-probabilities, so only these two losses apply
IMPLEMENTED_LOSSES = {"NLLLoss": NLLLoss,
                      "CrossEntropyLoss": CrossEntropyLoss}


def init_loss(loss_name: str, **kwargs):
    """Initialize the loss."""
    try:
        loss = IMPLEMENTED_LOSSES[loss_name](**kwargs)
    except KeyError as exc:
        raise NotImplementedError(
            f"{loss_name} is not implemented. " +
            f"Available Losses: {IMPLEMENTED_LOSSES.keys()}") from exc
    return loss
