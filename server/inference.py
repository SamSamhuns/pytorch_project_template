import os
import torch
import importlib
from PIL import Image
from collections import defaultdict


def init_detectors(models_param_dict: str, device='cuda:0'):
    print(f"Initializing detectors {models_param_dict.keys()}")

    models_dict = defaultdict(dict)
    for model_name, model_info in models_param_dict.items():
        # f_checkpoint = "path to checkpoint file .pth"
        # f_config = "path to model config or model.py itself"
        f_checkpoint, f_config = model_info
        det, det_config = init_single_detector(
            f_config, f_checkpoint, device=device)
        models_dict[model_name]['detector'], models_dict[model_name]['config'] = det, det_config
    return models_dict


def init_single_detector(f_config, f_checkpoint, device='cuda:0', type='pytorch'):
    print(f"Initializing detector from {f_checkpoint} with config {f_config}")
    if not os.path.exists(f_checkpoint) or not os.path.exists(f_config):
        raise RuntimeError(f"{f_checkpoint} and/or {f_config} does not exist!")
    if os.path.splitext(f_config)[1] != ".py":
        raise RuntimeError(f"{f_config} is not a .py file")

    f_config = f_config.replace('/', '.')[:-3]
    # import the nodel's config.py file
    config_module = importlib.import_module(f_config)
    detector = config_module.CONFIG["ARCH"]["TYPE"]()
    if type == "pytorch":
        detector = detector.to(device).eval()
    elif type == "onnx":
        raise NotImplementedError("onnx inference mode not implemented")
    detector_config = config_module.CONFIG
    return detector, detector_config


def inference_onnx():
    print("Running inference with onnx")
    raise NotImplementedError("onnx inference mode not implemented")


def inference_pytorch(model, preprocess_func, input_image_file, threshold):
    print("Running inference with pytorch")
    # read and preprocess input image depending on model here
    with torch.no_grad():
        pil_image = Image.open(input_image_file)
        preprocessed_tensor = preprocess_func(pil_image).unsqueeze(axis=0)
        preprocessed_tensor.requires_grad = False
        result = model(preprocessed_tensor)
        result = result.max(1, keepdim=True)[1].to("cpu").numpy()
    return result


def run_detector(model, preprocess_func, input_image_file: str, threshold: int = 0.55):
    print("Calling function run_detector")
    result = inference_pytorch(model, preprocess_func, input_image_file, threshold)
    name, ext = os.path.split(input_image_file)[1].split(".")
    name = name + "Gen." + ext
    # if results are to be drawn on image
    # out_img = os.path.join(os.path.dirname(input_image_file), name)
    # draw_results_on_image(input_image_file, result, out_file=out_img)
    return result


def draw_results_on_image(*args, **kwargs):
    print("Drawing results on Image")
