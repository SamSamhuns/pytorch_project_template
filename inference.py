import torch


def init_model(model_name: str, **kwargs):
    print("Initializing Detector")
    print("Calling function init_model")
    if models[model_name] is not None:
        return

    fCheckpoint = "path to checkpoint file .pth"
    fConfig = "path to model config or model.py itself"
    if not os.path.exists(fCheckpoint) or not os.path.exists(fConfig):
        raise RuntimeError(
            "directory: 'checkpoints' and/or 'configs' does not exist!")

    models[model_name] = init_detector(fConfig, fCheckpoint, device='cuda:0')
    return


def run_detector(input_model: InputModel, input_image_file: str, **kwargs):
    print("calling function run_detector")
    return "inference complete"
    model = models[input_model.model_name]

    result = inference_detector(model, input_image_file)
    name, ext = os.path.split(input_image_file)[1].split(".")
    name += "Gen."
    name += ext

    out_img = os.path.join(os.path.dirname(input_image_file), name)
    draw_results_on_image(input_image_file, result, out_file=out_img)
    return out_img


def draw_results_on_image(*args, **kwargs):
    print("Drawing results on Image")
