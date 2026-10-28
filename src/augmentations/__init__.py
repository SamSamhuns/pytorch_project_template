from collections.abc import Callable

from omegaconf import DictConfig

from .classifier_transforms import imagenet_transforms
from .mnist_transforms import mnist_transforms

# the dataset.preprocess.type values a config can name
TRANSFORMS = {"imagenet_transforms": imagenet_transforms, "mnist_transforms": mnist_transforms}


def build_transforms(cfg: DictConfig) -> tuple[Callable, Callable, dict]:
    """The train transform, eval transform and eval settings of cfg, the dataset.preprocess section of a config"""
    return TRANSFORMS[cfg.type](**cfg.args)
