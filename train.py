"""Training script, run with: python train.py --cfg YAML_CONFIG_PATH"""

import argparse
import os.path as osp
from datetime import datetime

import torch

from src.augmentations import build_transforms
from src.config_parser import CustomDictConfig
from src.datasets import DATASETS, build_dataloaders, log_label_counts
from src.loggers import setup_logging
from src.models import MODELS
from src.trainers import ClassifierTrainer
from src.utils.common import get_device, seed_everything


def get_config_from_args() -> CustomDictConfig:
    """Get CustomDictConfig obj from argparse"""
    parser = argparse.ArgumentParser(description="PyTorch Training. Currently only supports image classification")
    # primary cli args
    parser.add_argument(
        "--cfg",
        "--config",
        type=str,
        dest="config",
        default="configs/classifier_gpu_config.yaml",
        help="YAML config file path (default: %(default)s)",
    )
    parser.add_argument(
        "--id",
        "--run_id",
        type=str,
        dest="run_id",
        default="train_" + datetime.now().strftime(r"%Y%m%d_%H%M%S"),
        help="Unique identifier for training. Annotates train ckpts & logs. (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--override",
        type=str,
        nargs="+",
        dest="override",
        default=None,
        help="Override config params. Must match keys in YAML config. "
        "e.g. -o seed=1 dataset.type=NewDataType model.layers=[64,128,256] model.layers[2]=512 (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        default=False,
        help="Run training in verbose mode (default: %(default)s)",
    )

    # Add additional arguments here (Overrides YAML configs)
    parser.add_argument(
        "-r",
        "--resume_checkpoint",
        type=str,
        dest="resume_checkpoint",
        help="Checkpoint to resume training from, e.g. .../models/last.pth. "
        "Overrides `trainer.resume_checkpoint` in config.",
    )
    parser.add_argument("--dev", "--device", dest="device", help="Device to run on, e.g. cpu, cuda or cuda:1.")
    parser.add_argument(
        "--mode",
        type=str,
        dest="mode",
        default="TRAIN_TEST",
        choices=["TRAIN", "TRAIN_TEST"],
        help="Running mode. (default: %(default)s)",
    )
    args = parser.parse_args()

    # To override key-value params from YAML file,
    # match the YAML kv structure for any additional args above
    # keys-val pairs can have nested structure separated by colons
    yaml_modification = {
        "trainer.resume_checkpoint": args.resume_checkpoint,
        "device": args.device,
        "mode": args.mode,
    }
    # get omegaconf config obj
    return CustomDictConfig.from_args(args, yaml_modification)


def main():
    config = get_config_from_args()
    setup_logging(osp.join(config.logs_dir, "train.log"))
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
    if config.verbose:
        log_label_counts(dataset)
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset, config.dataloader, pin_memory=device.type == "cuda", generator=generator
    )

    model = MODELS[config.model.type](**config.model.args)
    run_dir = osp.join(config.save_dir, config.experiment_name, config.run_id)
    trainer = ClassifierTrainer(config, model, run_dir, device, classes=dataset.classes, verbose=config.verbose)
    if config.trainer.resume_checkpoint:
        trainer.load(config.trainer.resume_checkpoint)
    trainer.fit(train_loader, val_loader)
    # test this run's best.pth; it is None if this run saved none, then the last weights are tested
    if config.mode == "TRAIN_TEST" and test_loader is not None:
        trainer.evaluate(test_loader, weights=trainer.best_path)


if __name__ == "__main__":
    main()
