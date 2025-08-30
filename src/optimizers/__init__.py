from torch.optim import SGD, Adadelta, Adagrad, Adam, Adamax, AdamW, RMSprop

IMPLEMENTED_OPTIMIZERS = {"SGD": SGD, "Adam": Adam, "AdamW": AdamW, "Adadelta": Adadelta,
                          "Adagrad": Adagrad, "Adamax": Adamax, "RMSprop": RMSprop}


def init_optimizer(optimizer_name: str, **kwargs):
    """Initialize the optimizer."""
    try:
        optimizer = IMPLEMENTED_OPTIMIZERS[optimizer_name](**kwargs)
    except KeyError as exc:
        raise NotImplementedError(
            f"{optimizer_name} is not implemented. " +
            f"Available Optimizers: {IMPLEMENTED_OPTIMIZERS.keys()}") from exc
    return optimizer
