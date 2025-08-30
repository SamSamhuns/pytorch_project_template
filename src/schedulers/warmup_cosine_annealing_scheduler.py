import math
from functools import partial

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


def lr_warmup_cosine_annealing(epoch: int, total_epochs: int, warmup_epochs: int = 5) -> float:
    """Returns lr for given epoch after combining lr warmup and cosine annealing
    """
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        return 0.5 * (1 + math.cos((epoch - warmup_epochs) /
                                   (total_epochs - warmup_epochs) * math.pi))


class WarmupCosineAnnealingScheduler(LambdaLR):
    """Learning rate scheduler with warm-up and cosine annealing.

    Args:
        optimizer (Optimizer): The optimizer for which to adjust the learning rate.
        total_epochs (int): Total number of epochs for the main cosine annealing phase.
        warmup_epochs (int, optional): Number of epochs for the warm-up phase. Default is 5.
        **kwargs: Additional arguments to pass to the LambdaLR constructor.

    """

    def __init__(self, optimizer: Optimizer,
                 total_epochs: int, warmup_epochs: int = 5, **kwargs) -> None:
        lr_func = partial(lr_warmup_cosine_annealing,
                          total_epochs=total_epochs, warmup_epochs=warmup_epochs)
        super().__init__(optimizer, lr_lambda=lr_func, **kwargs)


if __name__ == "__main__":
    # test the lrwarmup cosine annealing scheduler
    import matplotlib.pyplot as plt
    from torch import nn
    from torch.optim import SGD

    model = nn.Linear(100, 10)
    optim = SGD(model.parameters(), lr=0.1)

    N_EPOCHS = 100
    scheduler = WarmupCosineAnnealingScheduler(
        optim, total_epochs=N_EPOCHS, warmup_epochs=5)

    epoch_list, lr_list = [], []
    for e in range(N_EPOCHS):
        optim.step()
        scheduler.step()

        lr = optim.param_groups[0]["lr"]
        e = (e+1)/N_EPOCHS
        epoch_list.append(e)
        lr_list.append(lr)

    plt.figure(figsize=(12, 8))
    plt.plot(epoch_list, lr_list)
    plt.title("LR warmup with cosine annelaing lr scheduler")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.show()
