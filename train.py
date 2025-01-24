"""
Training script, run with: python train.py --cfg YAML_CONFIG_PATH
"""
import argparse
from datetime import datetime

from src.config_parser import CustomDictConfig
from src.trainers import init_trainer


def get_config_from_args() -> CustomDictConfig:
    """Get CustomDictConfig obj from argparse"""
    parser = argparse.ArgumentParser(
        description='PyTorch Training. Currently only supports image classification')
    # primary cli args
    parser.add_argument(
        "--cfg", "--config", type=str, dest="config", default="configs/classifier_gpu_config.yaml",
        help="config file path (default: %(default)s)")
    parser.add_argument(
        "--id", "--run_id", type=str, dest="run_id", default="train_" + datetime.now().strftime(r"%Y%m%d_%H%M%S"),
        help="unique identifier for training. Annotates train ckpts & logs. (default: %(default)s)")
    parser.add_argument(
        "-o", "--override", type=str, nargs='+', dest="override", default=None,
        help="Override config params. Must match keys in YAML config. "
        "e.g. -o seed:1 dataset:type:DTYPE (default: %(default)s)")
    parser.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose", default=False,
        help="run training in verbose mode (default: %(default)s)")

    # Add additional arguments here (Overrides YAML configs)
    parser.add_argument(
        "-r", "--resume_checkpoint", type=str, dest="resume_checkpoint",
        help="Path to resume checkpoint. Overrides `trainer:resume_checkpoint` in config.")
    parser.add_argument(
        "--dev", "--device", dest="device", choices=["cpu", "cuda"],
        help="device for training. Use cpu or cuda.")
    parser.add_argument(
        "--gpu_device", type=int, dest="gpu_device", nargs="*",
        help="gpu_devices to use. Pass as space-sep numbers eg. --gpu_device 0 / 0 1 / 0 1 2.")
    parser.add_argument(
        "--mode", type=str, dest="mode", default="TRAIN_TEST",
        choices=["TRAIN", "TRAIN_TEST", "TRAIN_TEST_FEATSELECT"],
        help="Running mode. (default: %(default)s)")
    args = parser.parse_args()

    # To override key-value params from YAML file,
    # match the YAML kv structure for any additional args above
    # keys-val pairs can have nested structure separated by colons
    yaml_modification = {
        "trainer:resume_checkpoint": args.resume_checkpoint,
        "device": args.device,
        "gpu_device": args.gpu_device,
        "mode": args.mode
    }
    # get omegaconf config obj
    return CustomDictConfig.from_args(args, yaml_modification)


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
