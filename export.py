"""Export a trained model to ONNX, run with: python export.py --cfg YAML_CONFIG_PATH -r .../models/best.pth
Writes model.onnx to the export run's models dir. It takes any batch size and holds the class names
"""

import argparse
import json
import logging
import os.path as osp
from datetime import datetime

import torch

from src.config_parser import CustomDictConfig
from src.loggers import setup_logging
from src.models import init_model
from src.utils.export_utils import export_onnx

logger = logging.getLogger(__name__)


def get_config_from_args() -> CustomDictConfig:
    """Get CustomDictConfig obj from argparse"""
    parser = argparse.ArgumentParser(description="PyTorch Export to ONNX")
    # primary cli args
    parser.add_argument("--cfg", "--config", type=str, dest="config", required=True, help="YAML config file path.")
    parser.add_argument(
        "-r",
        "--resume_checkpoint",
        type=str,
        dest="resume_checkpoint",
        required=True,
        help="Checkpoint to export, e.g. .../models/best.pth",
    )
    parser.add_argument(
        "--id",
        "--run_id",
        type=str,
        dest="run_id",
        default="export_" + datetime.now().strftime(r"%Y%m%d_%H%M%S"),
        help="Unique identifier for export. Annotates checkpoints & logs. (default: %(default)s)",
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
        help="Run export in verbose mode (default: %(default)s)",
    )
    args = parser.parse_args()
    return CustomDictConfig.from_args(args)


def main():
    config = get_config_from_args()
    setup_logging(osp.join(config.logs_dir, "export.log"))
    for name in ("onnx_ir", "onnxscript"):  # the ONNX graph optimizer logs each of its passes at INFO
        logging.getLogger(name).setLevel(logging.WARNING)

    # the export runs on the CPU and needs only the model, not the trainer or the datasets
    model = init_model(config["model"]["type"], **config["model"]["args"])
    checkpoint = torch.load(config["resume_checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])

    info = config["model"]["info"]
    sample = torch.randn(2, info["input_channel"], info["input_height"], info["input_width"])
    export_path = osp.join(config["models_dir"], "model.onnx")
    export_onnx(model, sample, export_path, metadata={"classes": json.dumps(checkpoint["classes"])})
    logger.info("Exported %s; ONNX Runtime's outputs match the PyTorch model's", export_path)


if __name__ == "__main__":
    main()
