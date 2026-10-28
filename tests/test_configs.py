"""Every config in configs/ must build its model, transforms, loss, optimizer and LR scheduler"""

from pathlib import Path

import pytest
import torch
from PIL import Image

from src.augmentations import build_transforms
from src.config_parser import load_config
from src.metrics import METRICS, PLOTS
from src.models import MODELS
from src.trainers import ClassifierTrainer

CONFIGS = sorted((Path(__file__).parents[1] / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.name)
def test_config_builds(path, tmp_path):
    config = load_config(str(path), ["trainer.use_tensorboard=false"])
    model_args = dict(config.model.args)
    if "pretrained_weights" in model_args:
        model_args["pretrained_weights"] = None  # offline; the weights don't change the model's shapes
    model = MODELS[config.model.type](**model_args).eval()
    _, eval_transform, _ = build_transforms(config.dataset.preprocess)

    with torch.inference_mode():  # the model takes the eval transform's output, e.g. one channel for MNIST
        logits = model(eval_transform(Image.new("RGB", (64, 48))).unsqueeze(0))
    assert logits.shape == (1, model_args["num_classes"])
    ClassifierTrainer(config, model, str(tmp_path), torch.device("cpu"))  # the loss, optimizer and LR scheduler
    assert set(config.metrics.val) | set(config.metrics.test) <= METRICS.keys() | PLOTS.keys()
