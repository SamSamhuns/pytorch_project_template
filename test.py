"""Test a trained model on the test split, run with: python test.py --cfg CONFIG -w .../models/best.pth"""

import argparse
import os.path as osp

from omegaconf import OmegaConf

from src.augmentations import build_transforms
from src.config_parser import add_common_args, load_config, make_run_dir
from src.datasets import DATASETS, build_dataloaders
from src.loggers import setup_logging
from src.models import MODELS
from src.trainers import ClassifierTrainer
from src.utils.common import get_device


def main():
    parser = argparse.ArgumentParser(description="Test a trained image classifier on the test split")
    add_common_args(parser, run_prefix="test")
    parser.add_argument("-w", "--weights", required=True, help="Checkpoint to test, e.g. .../models/best.pth")
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    run_dir = make_run_dir(config, args.run_id)
    setup_logging(osp.join(run_dir, "logs", "test.log"))
    OmegaConf.save(config, osp.join(run_dir, "config.yaml"))
    device = get_device(config.device)

    train_transform, eval_transform, _ = build_transforms(config.dataset.preprocess)
    dataset = DATASETS[config.dataset.type](
        train_transform=train_transform, eval_transform=eval_transform, **config.dataset.args
    )
    _, _, test_loader = build_dataloaders(dataset, config.dataloader, pin_memory=device.type == "cuda")
    if test_loader is None:
        raise ValueError("The dataset has no test split")

    model = MODELS[config.model.type](**config.model.args)
    trainer = ClassifierTrainer(config, model, run_dir, device)
    trainer.evaluate(test_loader, weights=args.weights)


if __name__ == "__main__":
    main()
