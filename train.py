"""Train a model, run with: python train.py --cfg configs/mnist_config.yaml
Saves last.pth every epoch and best.pth whenever the val accuracy improves, then tests best.pth on the test split
"""

import argparse
import logging
import os.path as osp

import torch
from omegaconf import OmegaConf

from src.augmentations import build_transforms
from src.config_parser import add_common_args, load_config, make_run_dir
from src.datasets import DATASETS, build_dataloaders, log_label_counts
from src.loggers import setup_logging
from src.models import MODELS
from src.trainers import ClassifierTrainer
from src.utils.common import get_device, get_git_revision_hash, seed_everything

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train an image classifier")
    add_common_args(parser, run_prefix="train")
    parser.add_argument("-r", "--resume", help="Checkpoint to resume training from, e.g. .../models/last.pth")
    parser.add_argument("-v", "--verbose", action="store_true", help="Also log the label counts and batch losses")
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    run_dir = make_run_dir(config, args.run_id)
    setup_logging(osp.join(run_dir, "logs", "train.log"))
    OmegaConf.save(config, osp.join(run_dir, "config.yaml"))
    logger.info("Git commit: %s", get_git_revision_hash())
    if config.seed is not None:
        seed_everything(config.seed)
    # reproducible runs trade speed for determinism; otherwise cuDNN benchmarks and picks the fastest algorithms
    torch.backends.cudnn.benchmark = not config.reproducible
    if config.reproducible:
        torch.use_deterministic_algorithms(True, warn_only=True)  # warns on ops that have no deterministic version
    device = get_device(config.device)

    # a seeded generator makes the val split and the train shuffle reproducible
    generator = torch.Generator().manual_seed(config.seed) if config.seed is not None else None
    train_transform, eval_transform, _ = build_transforms(config.dataset.preprocess)
    dataset = DATASETS[config.dataset.type](
        train_transform=train_transform, eval_transform=eval_transform, generator=generator, **config.dataset.args
    )
    if args.verbose:
        log_label_counts(dataset)
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset, config.dataloader, pin_memory=device.type == "cuda", generator=generator
    )

    model = MODELS[config.model.type](**config.model.args)
    trainer = ClassifierTrainer(config, model, run_dir, device, classes=dataset.classes, verbose=args.verbose)
    if args.resume:
        trainer.load(args.resume)
    trainer.fit(train_loader, val_loader)
    if test_loader is not None:
        # this run's best.pth; it is None if this run saved none, and then the last weights are tested
        trainer.evaluate(test_loader, weights=trainer.best_path)


if __name__ == "__main__":
    main()
