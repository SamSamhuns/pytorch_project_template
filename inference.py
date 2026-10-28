"""Script for running inference with trained .pt torch models

Note: The inference script can be ported for use without a config file as well.
Use the torch checkpoint file and the necessary transforms that are used for porting
"""

import argparse
import os.path as osp
from datetime import datetime

from src.config_parser import CustomDictConfig
from src.loggers import setup_logging
from src.trainers import init_trainer


def get_config_from_args():
    parser = argparse.ArgumentParser(description="PyTorch Inference. Currently only supports image classification")
    # primary cli args
    parser.add_argument("--cfg", "--config", type=str, dest="config", required=True, help="YAML config file path.")
    parser.add_argument(
        "-r",
        "--resume_checkpoint",
        type=str,
        dest="resume_checkpoint",
        required=True,
        help="Path to resume checkpoint. Overrides `trainer.resume_checkpoint` in config.",
    )
    parser.add_argument(
        "--id",
        "--run_id",
        type=str,
        dest="run_id",
        default="test_" + datetime.now().strftime(r"%Y%m%d_%H%M%S"),
        help="Unique identifier for testing. Annotates checkpoints & logs. (default: %(default)s)",
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
        help="Run testing in verbose mode (default: %(default)s)",
    )

    # additional arguments (Overrides YAML configs)
    parser.add_argument("--dev", "--device", dest="device", help="Device to run on, e.g. cpu, cuda or cuda:1.")
    parser.add_argument(
        "--mode",
        type=str,
        dest="mode",
        default="INFERENCE",
        choices=["INFERENCE"],
        help="Running mode. (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--source_path",
        type=str,
        dest="source_path",
        required=True,
        help="Path to source image file or directory for running inference.",
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
    return CustomDictConfig.from_args(args, yaml_modification)


def main():
    config = get_config_from_args()
    setup_logging(osp.join(config.logs_dir, "inference.log"))
    config["trainer"]["use_tensorboard"] = False
    trainer = init_trainer(config["trainer"]["type"], config=config, logger_name="inference")

    trainer.inference(config["source_path"])


if __name__ == "__main__":
    main()
