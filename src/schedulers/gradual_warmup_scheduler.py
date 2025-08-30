# Gradual Warmup Scheduler
# Implemented from Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour https://arxiv.org/pdf/1706.02677.pdf
# Gradual increase in learning rate by a constant amount to avoid sudden increase in lr

import warnings

from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


class GradualWarmupScheduler(_LRScheduler):
    """Sets the learning rate of parameter group to gradually increase for num_epochs from start_lr
    to the original lr set for the optimizer

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        eps_lr (float): min or starting learning rate which is gradually/linearly increased
            to optimizer lr. Default: 0.000001
        warmup_epochs (int): num of epochs during which the lr is increased. Default: 5.
        after_scheduler (Scheduler): scheduler to use after gradual warmup of lr is done. Default: None.
        last_epoch (int): The index of last epoch. Default: -1.
            verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    """

    def __init__(self, optimizer, eps_lr=0.000001, warmup_epochs=5, after_scheduler=None, last_epoch=-1, verbose=False):
        self.warmup_epochs = warmup_epochs
        self.eps_lr = eps_lr
        self.after_scheduler = after_scheduler

        get_last_lr = getattr(self.after_scheduler, "get_last_lr", None)
        if not callable(get_last_lr):
            def _get_last_lr():
                return [group['lr'] for group in self.optimizer.param_groups]
            get_last_lr = _get_last_lr
        self.after_scheduler.get_last_lr = get_last_lr

        self.finished = False  # set to True when warmup done
        super().__init__(
            optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch > self.warmup_epochs:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr for base_lr in self.base_lrs]

        return [max(base_lr * (self.last_epoch / self.warmup_epochs), self.eps_lr)
                for base_lr in self.base_lrs]

    def step(self, metrics=None, epoch=None):
        # metrics is discarded unless ReduceLROnPlateau is used as after_scheduler
        # adopted from official pytorch _LRScheduler implementation
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1
        if self.finished and self.after_scheduler:
            # if ReduceLROnPlateau is used, use the metrics parameter
            if isinstance(self.after_scheduler, ReduceLROnPlateau):
                return self.after_scheduler.step(metrics=metrics, epoch=epoch)
            if epoch is None:
                self.after_scheduler.step()
            else:
                self.after_scheduler.step(epoch=epoch - self.warmup_epochs)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(GradualWarmupScheduler, self).step()


def main():
    # for testing the custom scheduler only
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import tqdm
    from torch.optim import SGD
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, 256),
                nn.ReLU(),
                nn.Linear(256, 10),
                nn.ReLU()
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return F.log_softmax(logits, dim=1)

    def train(model, x, y, optimizer, criterion):
        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        return loss, output

    device = "cpu"
    EPOCHS = 100
    BATCH_SIZE = 32
    LR = 0.001
    net = Network().to(device)
    optimizer = SGD(net.parameters(), lr=LR)
    # after_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    after_scheduler = ReduceLROnPlateau(optimizer, patience=5)
    criterion = nn.NLLLoss()
    scheduler = GradualWarmupScheduler(
        optimizer, after_scheduler=after_scheduler)

    # ####### sample scheduler test ####### #
    # for epoch in range(EPOCHS):
    #     print(f"epoch {epoch}: lr={optimizer.param_groups[0]['lr']}")
    #     scheduler.step(metrics=2)
    # exit()

    # ####### full scheduler test with mnist ######## #
    mnist_train_data = datasets.MNIST("data",
                                      train=True,
                                      download=True,
                                      transform=transforms.Compose([transforms.ToTensor(),
                                                                    transforms.Normalize((0.1307,), (0.3081,))]))
    data_train_len = len(mnist_train_data) // 40
    sample_list = [i for i in range(data_train_len)]
    trainset_subset = Subset(mnist_train_data, sample_list)
    data_train = DataLoader(trainset_subset, batch_size=BATCH_SIZE,
                            shuffle=True)

    train_accuracy, train_loss, train_lr = [], [], []
    for epoch in range(EPOCHS):
        epoch_loss = 0
        correct = 0
        for batch in tqdm.tqdm(data_train):
            x_train, y_train = batch
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            loss, predictions = train(
                net, x_train, y_train, optimizer, criterion)
            for idx, i in enumerate(predictions):
                i = torch.round(i).detach().numpy()
                pred = np.argmax(i)
                if pred == y_train[idx]:
                    correct += 1
            acc = (correct / data_train_len)
            epoch_loss += loss
        train_accuracy.append(acc * 100)
        train_loss.append(epoch_loss.detach().numpy() ** (1 / 2))
        train_lr.append(optimizer.param_groups[0]["lr"])
        # step scheduler on each epoch instead of batch
        scheduler.step(metrics=(1 - acc))
        print('Epoch: {} LR {} Accuracy: {}, Loss: {}'.format(
            epoch + 1, optimizer.param_groups[0]["lr"], acc * 100, epoch_loss))

    epoch_list = [i for i in range(EPOCHS)]

    # plot the leraing rate along with training accuracy over epochs
    _, ax = plt.subplots()
    plt.title(f"Train with BSIZE {BATCH_SIZE}")
    ax.plot(epoch_list, train_accuracy, color="red")
    ax.set_xlabel("EPOCHS", fontsize=14)
    ax.set_ylabel("accuracy", color="red", fontsize=14)

    ax2 = ax.twinx()
    ax2.plot(epoch_list, train_lr, color="blue")
    ax2.set_ylabel("learning rate", color="blue", fontsize=8)

    plt.savefig(
        f'train_stats_bsize_{BATCH_SIZE}_lr_{LR}.png', format="jpg")


if __name__ == "__main__":
    main()
