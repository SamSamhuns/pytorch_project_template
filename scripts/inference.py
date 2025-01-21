"""
Script for running inference with trained .pt torch models

Note: The inference script can be ported for use without a config file as well.
Use the torch checkpoint file and the necessary transforms that are used for porting
"""
import argparse
from datetime import datetime

from src.trainers import init_trainer
from src.config_parser import CustomDictConfig


def get_config_from_args():
    parser = argparse.ArgumentParser(
        description="PyTorch Inference. Currently only supports image classification")
    # primary cli args
    parser.add_argument(
        "--cfg", "--config", type=str, dest="config", required=True,
        help="config file path (default: %(default)s)")
    parser.add_argument(
        "--id", "--run_id", type=str, dest="run_id", default="inference_" + datetime.now().strftime(r"%Y%m%d_%H%M%S"),
        help="unique identifier for inference process when saving results. (default: %(default)s)")
    parser.add_argument(
        "-r", "--resume", type=str, dest="resume", required=True,
        help="path to resume ckpt for running inference.")
    parser.add_argument(
        "-o", "--override", type=str, nargs="+", dest="override", default=None,
        help="Override config params. Must match keys in YAML config. "
        "e.g. -o seed:1 dataset:type:DTYPE (default: %(default)s)")
    parser.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose", default=False,
        help="run training in verbose mode (default: %(default)s)")

    # additional custom cli options that ovveride or add new config params from YAML config file
    override_options = [
        {"flags": ["-s", "--source_path"],
         "dest": "source_path", "required": True,
         "help": "path to source image file or directory for running inference. (default: %(default)s)",
         "type": str, "target": "source_path"},
        {"flags": ["--dev", "--gpu_device"],
         "dest": "gpu_device",
         "help": "gpu_device list eg. 0, 0 1, 0 1 2. Pass --dev with no arg for cpu (default: %(default)s)",
         "nargs": "*",
         "type": int, "target": "gpu_device"},
        {"flags": ["--mode"],
         "dest": "mode", "default": "INFERENCE", "choices": ["INFERENCE"],
         "help": "Running mode. Cannot be changed & fixed to INFERENCE (default: %(default)s)",
         "type": str, "target": "mode"}
    ]
    config = CustomDictConfig.from_args(parser, override_options)
    return config


def main():
    config = get_config_from_args()
    config["trainer"]["use_tensorboard"] = False
    trainer = init_trainer(
        config["trainer"]["type"], config=config, logger_name="inference")

    trainer.inference(config["source_path"])


if __name__ == "__main__":
    main()
