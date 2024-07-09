import sys
sys.path.append("./")
import os
import torch
from typing import Optional

from src.agents import classifier_agent
from src.config_parser import ConfigParser
from src.utils.common import read_json


def init_detectors(models_param_dict: dict, device: Optional[str] = "0"):
    """
    Args:
        models_param_dict: dict = dict key (model name): val ([model weight, model config])
        device: str = device where inf is run, "cpu", "0", "0,1", "0,1,2"
    """
    print(f"Initializing models {models_param_dict.keys()} in device {device}")

    models_dict = dict()
    for model_name, model_info in models_param_dict.items():
        f_checkpoint, f_config = model_info
        agent = init_single_detector(f_config, f_checkpoint, device=device)
        models_dict[model_name] = agent
    return models_dict


def init_single_detector(f_config: str, f_checkpoint: str, device: Optional[str]="0", inf_type: str = "pytorch"):
    """
    Args:
        f_config: str = path to json config file
        f_checkpoint: str = path to checkpoint .pth file
        device: str = device where inf is run, "cpu", "0", "0,1", "0,1,2"
        inf_type: str = pytorch/onnx
    """
    print(f"Initializing detector from {f_checkpoint} with config {f_config}")
    if not os.path.exists(f_checkpoint) or not os.path.exists(f_config):
        raise RuntimeError(f"{f_checkpoint} and/or {f_config} does not exist!")

    dev = None if ("cpu" in device or device is None) else list(map(int, device.split(',')))
    config = ConfigParser(config=read_json(f_config), resume=f_checkpoint, modification={"gpu_device": dev, "mode": "INFERENCE"})
    agent = classifier_agent.ClassifierAgent(config, "inference")

    if inf_type == "pytorch":
        print("Initializing pytorch inference mode")
    if inf_type == "onnx":
        raise NotImplementedError("onnx inference mode not implemented")
    return agent


def inference_onnx():
    print("Running inference with onnx")
    raise NotImplementedError("onnx inference mode not implemented")


def inference_pytorch(model, input_image_file, threshold):
    print("Running inference with pytorch")
    # read and preprocess input image depending on model here
    # TODO fix the inference function call since we need to pass the weight path
    with torch.no_grad():
        _, pred = model.inference(input_image_file, log_txt_preds=False)[0]
    return pred


def run_detector(model, input_image_file: str, threshold: int = 0.55):
    print("Calling function run_detector")
    pred = inference_pytorch(model, input_image_file, threshold)
    name, ext = os.path.split(input_image_file)[1].split(".")
    name = name + "Gen." + ext
    # if results are to be drawn on image
    # out_img = os.path.join(os.path.dirname(input_image_file), name)
    # draw_results_on_image(input_image_file, result, out_file=out_img)
    return pred


def draw_results_on_image(*args, **kwargs):
    print("Drawing results on Image")
