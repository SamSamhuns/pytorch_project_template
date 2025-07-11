"""
Export script to convert pytorch model to torshscript/onnx.
Run with: python export.py --cfg YAML_CONFIG_PATH -r PTH_MODEL_PATH --mode ONNX_TS
"""
import argparse
from datetime import datetime

from src.config_parser import CustomDictConfig
from src.trainers import init_trainer


def get_config_from_args() -> CustomDictConfig:
    """Get CustomDictConfig obj from argparse"""
    parser = argparse.ArgumentParser(
        description="PyTorch Export. Supports ONNX and TorchScript export")
    # primary cli args
    parser.add_argument(
        "--cfg", "--config", type=str, dest="config", required=True,
        help="YAML config file path.")
    parser.add_argument(
        "-r", "--resume_checkpoint", type=str, dest="resume_checkpoint", required=True,
        help="Path to resume checkpoint. Overrides `trainer.resume_checkpoint` in config.")
    parser.add_argument(
        "--id", "--run_id", type=str, dest="run_id", default="export_" + datetime.now().strftime(r"%Y%m%d_%H%M%S"),
        help="Unique identifier for export. Annotates checkpoints & logs. (default: %(default)s)")
    parser.add_argument(
        "-o", "--override", type=str, nargs="+", dest="override", default=None,
        help="Override config params. Must match keys in YAML config. "
        "e.g. -o seed=1 dataset.type=NewDataType model.layers=[64,128,256] model.layers[2]=512 (default: %(default)s)")
    parser.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose", default=False,
        help="Run export in verbose mode (default: %(default)s)")

    # additional arguments (Overrides YAML configs)
    parser.add_argument(
        "--dev", "--device", dest="device", choices=["cpu", "cuda"],
        help="device for training. Use cpu or cuda.")
    parser.add_argument(
        "--gpu_device", type=int, dest="gpu_device", nargs="*",
        help="gpu_devices to use. Pass as space-sep numbers eg. --gpu_device 0 / 0 1 / 0 1 2.")
    parser.add_argument(
        "--mode", type=str, dest="mode", required=True,
        choices=["ONNX_TS", "ONNX_DYNAMO", "TS_TRACE", "TS_SCRIPT"],
        help="Running mode.")
    parser.add_argument(
        "-q", "--quant_backend", type=str, dest="quant_backend",
        help="Quantization mode backend. (If None, dont quantize. Only supports TS_SCRIPT)",
        choices=["fbgemm", "x86", "qnnpack", "onednn"])
    args = parser.parse_args()

    # To override key-value params from YAML file,
    # match the YAML kv structure for any additional args above
    # keys-val pairs can have nested structure separated by colons
    yaml_modification = {
        "trainer.resume_checkpoint": args.resume_checkpoint,
        "device": args.device,
        "gpu_device": args.gpu_device,
        "mode": args.mode,
        "quant_backend": args.quant_backend,
    }
    return CustomDictConfig.from_args(args, yaml_modification)


def main():
    config = get_config_from_args()
    trainer = init_trainer(
        config["trainer"]["type"], config=config, logger_name="export")

    trainer.export(config["mode"], config["quant_backend"])


if __name__ == "__main__":
    main()
