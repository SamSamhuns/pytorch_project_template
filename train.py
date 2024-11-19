"""
Training script, run with: python train.py --cfg JSON_CONFIG_PATH
"""
import argparse
from datetime import datetime

from src.config_parser import ConfigParser
from src.trainers import init_trainer


def get_config_from_args() -> ConfigParser:
    """Get ConfigParser obj from argparse"""
    parser = argparse.ArgumentParser(
        description='PyTorch Training. Currently only supports image classification')
    # primary cli args
    parser.add_argument(
        "--cfg", "--config", type=str, dest="config", default="configs/classifier_cpu_config.json",
        help="config file path (default: %(default)s)")
    parser.add_argument(
        "--id", "--run_id", type=str, dest="run_id", default="train_" + datetime.now().strftime(r"%Y%m%d_%H%M%S"),
        help="unique identifier for training. Annotates train ckpts & logs. (default: %(default)s)")
    parser.add_argument(
        "-r", "--resume", type=str, dest="resume", default=None,
        help="path to resume ckpt. Overrides `resume_checkpoint` in config. (default: %(default)s)")
    parser.add_argument(
        "-o", "--override", type=str, nargs='+', dest="override", default=None,
        help="Override config params. Must match keys in json config. "
        "e.g. -o seed:1 dataset:type:DTYPE (default: %(default)s)")
    parser.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose", default=False,
        help="run training in verbose mode (default: %(default)s)")

    # custom cli options to modify cfg from values given in the json cfg file.
    # works in the same way as the -o/--override argument above, which takes precendent
    # to change LR {"flags": ["--lr"], "dest": "lr", "type": float, "target": "optimizer:args:lr"}
    override_options = [
        {"flags": ["--dev", "--gpu_device"],
         "dest": "gpu_device",
         "help": "gpu_device list eg. 0, 0 1, 0 1 2. Pass --dev with no arg for cpu (default: %(default)s)",
         "nargs": "*",
         "type": int, "target": "gpu_device"},
        {"flags": ["--mode"],
         "dest": "mode",
         "help": "Running mode. (default: %(default)s)",
         "default": "TRAIN_TEST", "choices": ["TRAIN", "TRAIN_TEST", "TRAIN_TEST_FEATSELECT"],
         "type": str, "target": "mode"}
    ]
    return ConfigParser.from_args(parser, override_options)


def main():
    config = get_config_from_args()
    trainer = init_trainer(config["trainer"]["type"], config=config, logger_name="train")

    trainer.train()
    if config["mode"] == "TRAIN_TEST":
        trainer.test()
    elif config["mode"] == "TRAIN_TEST_FEATSELECT":
        trainer.test()
        trainer.calc_feature_importance()


if __name__ == "__main__":
    main()
