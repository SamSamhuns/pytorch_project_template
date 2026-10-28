"""Export a trained model to ONNX, run with: python export.py --cfg YAML_CONFIG_PATH -w .../models/best.pth
Writes model.onnx to the export run's models dir. It takes any batch size, and holds the class names and
the settings of the eval transform, which server/app.py reads
"""

import argparse
import json
import logging
import os.path as osp

import torch
from omegaconf import OmegaConf

from src.augmentations import build_transforms
from src.config_parser import add_common_args, load_config, make_run_dir
from src.loggers import setup_logging
from src.models import MODELS
from src.utils.export_utils import export_onnx

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export a trained model to ONNX")
    add_common_args(parser, run_prefix="export")
    parser.add_argument("-w", "--weights", required=True, help="Checkpoint to export, e.g. .../models/best.pth")
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    run_dir = make_run_dir(config, args.run_id)
    setup_logging(osp.join(run_dir, "logs", "export.log"))
    OmegaConf.save(config, osp.join(run_dir, "config.yaml"))
    for name in ("onnx_ir", "onnxscript"):  # the ONNX graph optimizer logs each of its passes at INFO
        logging.getLogger(name).setLevel(logging.WARNING)

    # the export runs on the CPU and needs only the model, not the trainer or the datasets
    model = MODELS[config.model.type](**config.model.args)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    _, _, eval_settings = build_transforms(config.dataset.preprocess)

    # a batch of 2 random images, with the channels and size of the eval transform's output
    channels, size = {"L": 1, "RGB": 3}[eval_settings["image_mode"]], eval_settings["crop_size"]
    sample = torch.randn(2, channels, size, size)
    export_path = osp.join(run_dir, "models", "model.onnx")
    metadata = {"classes": json.dumps(checkpoint["classes"]), "preprocess": json.dumps(eval_settings)}
    export_onnx(model, sample, export_path, metadata)
    logger.info("Exported %s; ONNX Runtime's outputs match the PyTorch model's", export_path)


if __name__ == "__main__":
    main()
