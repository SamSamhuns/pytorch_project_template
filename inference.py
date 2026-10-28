"""Inference on an image file or a dir of images,
run with: python inference.py --cfg YAML_CONFIG_PATH -w .../models/best.pth -s IMAGE_OR_DIR
Logs the most likely class of each image with its probability, and writes them to the run's logs/pred.txt
"""

import argparse
import glob
import logging
import os.path as osp

import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.datasets.folder import is_image_file

from src.augmentations import build_transforms
from src.config_parser import add_common_args, load_config, make_run_dir
from src.loggers import setup_logging
from src.models import MODELS
from src.utils.common import get_device

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Classify images with a trained model")
    add_common_args(parser, run_prefix="inference")
    parser.add_argument("-w", "--weights", required=True, help="Checkpoint to run, e.g. .../models/best.pth")
    parser.add_argument("-s", "--source_path", required=True, help="Image file or dir of images to classify")
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    run_dir = make_run_dir(config, args.run_id)
    setup_logging(osp.join(run_dir, "logs", "inference.log"))
    OmegaConf.save(config, osp.join(run_dir, "config.yaml"))
    device = get_device(config.device)

    # inference needs only the model and the eval transform, not the trainer or the datasets
    model = MODELS[config.model.type](**config.model.args)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    classes = checkpoint["classes"]
    _, eval_transform, _ = build_transforms(config.dataset.preprocess)

    # is_image_file compares lower-cased extensions, so .JPG and .PNG files count too
    source_path = args.source_path
    if osp.isdir(source_path):
        image_paths = sorted(path for path in glob.glob(osp.join(source_path, "*")) if is_image_file(path))
    elif osp.isfile(source_path) and is_image_file(source_path):
        image_paths = [source_path]
    else:
        raise ValueError(f"Inference source {source_path} is not an image file or dir with images")

    with torch.inference_mode(), open(osp.join(run_dir, "logs", "pred.txt"), "w", encoding="utf-8") as pred_file:
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
