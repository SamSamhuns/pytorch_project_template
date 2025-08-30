from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, StepLR

from .warmup_cosine_annealing_scheduler import WarmupCosineAnnealingScheduler

IMPLEMENTED_SCHEDULERS = {
    "StepLR": StepLR,
    "OneCycleLR": OneCycleLR,
    "ReduceLROnPlateau": ReduceLROnPlateau,
    "WarmupCosineAnnealingScheduler": WarmupCosineAnnealingScheduler
}


def init_scheduler(scheduler_name: str, **kwargs):
    """Initialize the scheduler."""
    try:
        scheduler = IMPLEMENTED_SCHEDULERS[scheduler_name](**kwargs)
    except KeyError as exc:
        raise NotImplementedError(
            f"{scheduler_name} is not implemented. " +
            f"Available Scheduler: {IMPLEMENTED_SCHEDULERS.keys()}") from exc
    return scheduler
