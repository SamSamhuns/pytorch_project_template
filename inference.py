"""Inference on an image file or a dir of images,
run with: python inference.py --cfg YAML_CONFIG_PATH -r .../models/best.pth -s IMAGE_OR_DIR
Logs the most likely class of each image with its probability, and writes them to the run's logs/pred.txt
"""

import argparse
import glob
import logging
import os.path as osp
from datetime import datetime

import torch
from PIL import Image
from torchvision.datasets.folder import is_image_file

from src.augmentations import build_transforms
from src.config_parser import CustomDictConfig
from src.loggers import setup_logging
from src.models import MODELS
from src.utils.common import get_device

logger = logging.getLogger(__name__)


def get_config_from_args() -> CustomDictConfig:
    """Get CustomDictConfig obj from argparse"""
    parser = argparse.ArgumentParser(description="PyTorch Inference. Currently only supports image classification")
    # primary cli args
    parser.add_argument("--cfg", "--config", type=str, dest="config", required=True, help="YAML config file path.")
    parser.add_argument(
        "-r",
        "--resume_checkpoint",
        type=str,
        dest="resume_checkpoint",
        required=True,
        help="Checkpoint to run, e.g. .../models/best.pth",
    )
    parser.add_argument(
        "--id",
        "--run_id",
        type=str,
        dest="run_id",
        default="inference_" + datetime.now().strftime(r"%Y%m%d_%H%M%S"),
        help="Unique identifier for inference. Annotates logs and predictions. (default: %(default)s)",
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
        help="Run inference in verbose mode (default: %(default)s)",
    )

    # additional arguments (Overrides YAML configs)
    parser.add_argument("--dev", "--device", dest="device", help="Device to run on, e.g. cpu, cuda or cuda:1.")
    parser.add_argument(
        "-s",
        "--source_path",
        type=str,
        dest="source_path",
        required=True,
        help="Path to source image file or directory for running inference.",
    )
    args = parser.parse_args()
    return CustomDictConfig.from_args(args, {"device": args.device})


def main():
    config = get_config_from_args()
    setup_logging(osp.join(config.logs_dir, "inference.log"))
    device = get_device(config["device"])

    # inference needs only the model and the eval transform, not the trainer or the datasets
    model = MODELS[config["model"]["type"]](**config["model"]["args"])
    checkpoint = torch.load(config["resume_checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    classes = checkpoint["classes"]
    _, eval_transform, _ = build_transforms(config["dataset"]["preprocess"])

    # is_image_file compares lower-cased extensions, so .JPG and .PNG files count too
    source_path = config["source_path"]
    if osp.isdir(source_path):
        image_paths = sorted(path for path in glob.glob(osp.join(source_path, "*")) if is_image_file(path))
    elif osp.isfile(source_path) and is_image_file(source_path):
        image_paths = [source_path]
    else:
        raise ValueError(f"Inference source {source_path} is not an image file or dir with images")

    with torch.inference_mode(), open(osp.join(config["logs_dir"], "pred.txt"), "w", encoding="utf-8") as pred_file:
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            probs = model(eval_transform(image).unsqueeze(0).to(device)).softmax(dim=1)[0]
            prob, pred = probs.max(dim=0)
            class_name, prob = classes[pred.item()], prob.item()
            logger.info("%s: %s (probability %.3f)", image_path, class_name, prob)
            pred_file.write(f"{image_path}, {class_name}, {prob:.4f}\n")
    logger.info("Inference complete for %s", source_path)


if __name__ == "__main__":
    main()
