"""
Export script to convert pytorch model to torshscript/onnx.
Run with: python export.py --cfg JSON_CONFIG_PATH -r PTH_MODEL_PATH --mode ONNX_TS
"""
import argparse
from datetime import datetime

from src.config_parser import ConfigParser
from src.trainers import init_trainer


def get_config_from_args() -> ConfigParser:
    """Get ConfigParser obj from argparse"""
    parser = argparse.ArgumentParser(
        description="PyTorch Training. Supports time series classification & anomaly detection")
    # primary cli args
    parser.add_argument(
        "--cfg", "--config", type=str, dest="config", required=True,
        help="config file path")
    parser.add_argument(
        "--id", "--run_id", type=str, dest="run_id", default="export_" + datetime.now().strftime(r"%Y%m%d_%H%M%S"),
        help="unique identifier for testing. Annotates export ckpts & logs. (default: %(default)s)")
    parser.add_argument(
        "-r", "--resume", type=str, dest="resume", required=True,
        help="path to resume ckpt. Overrides `resume_checkpoint` in config.")
    parser.add_argument(
        "-o", "--override", type=str, nargs="+", dest="override", default=None,
        help="Override config params. Must match keys in json config. "
        "e.g. -o seed:1 dataset:type:DTYPE (default: %(default)s)")
    parser.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose", default=False,
        help="run training in verbose mode (default: %(default)s)")

    # custom cli options to modify cfg from values given in the json cfg file.
    # works in the same way as the -o/--override argument above, which takes precendent

    # custom cli options to modify cfg from values given in the json cfg file.
    override_options = [
        {"flags": ["--dev", "--gpu_device"],
         "dest": "gpu_device",
         "help": "gpu_device list eg. 0, 0 1, 0 1 2. Pass --dev with no arg for cpu (default: %(default)s)",
         "nargs": "*",
         "type": int, "target": "gpu_device"},
        {"flags": ["-q", "--quant_backend"],
         "dest": "quant_backend",
         "help": "Quantization mode backend. (If None, dont quantize. Only supports TS_SCRIPT)",
         "choices": ["fbgemm", "x86", "qnnpack", "onednn"],
         "type": str, "target": "quant_backend"},
        {"flags": ["--mode"],
         "dest": "mode",
         "help": "Export mode.",
         "required": True, "choices": ["ONNX_TS", "ONNX_DYNAMO", "TS_TRACE", "TS_SCRIPT"],
         "type": str, "target": "mode"}
    ]

    return ConfigParser.from_args(parser, override_options)


def main():
    config = get_config_from_args()
    trainer = init_trainer(
        config["trainer"]["type"], config=config, logger_name="export")

    trainer.export(config["mode"], config["quant_backend"])


if __name__ == "__main__":
    main()
