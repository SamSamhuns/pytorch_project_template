from .base_trainer import BaseTrainer
from .classifier_trainer import ClassifierTrainer


IMPLEMENTED_TRAINERS = {
    "ClassifierTrainer": ClassifierTrainer,
}


def init_trainer(trainer_name: str, **kwargs):
    """Initialize the trainer."""
    try:
        trainer = IMPLEMENTED_TRAINERS[trainer_name](**kwargs)
    except KeyError as exc:
        raise NotImplementedError(
            f"{trainer_name} is not implemented. " +
            f"Available Trainers: {IMPLEMENTED_TRAINERS.keys()}") from exc
    return trainer
