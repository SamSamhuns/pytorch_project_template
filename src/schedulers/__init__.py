from omegaconf import DictConfig
from torch.optim import Optimizer, lr_scheduler


def init_scheduler(optimizer: Optimizer, cfg: DictConfig) -> lr_scheduler.LRScheduler:
    """Build any scheduler in torch.optim.lr_scheduler from cfg.type and cfg.args.
    With cfg.warmup_epochs, the LR first climbs linearly from 10% to 100% (not with ReduceLROnPlateau).
    """
    scheduler = getattr(lr_scheduler, cfg["type"])(optimizer, **cfg["args"])
    warmup_epochs = cfg.get("warmup_epochs", 0)
    if warmup_epochs:
        warmup = lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        scheduler = lr_scheduler.SequentialLR(optimizer, [warmup, scheduler], milestones=[warmup_epochs])
    return scheduler
