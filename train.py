"""
Training script, run with: python train.py --cfg JSON_CONFIG_PATH
"""
import argparse
from datetime import datetime

import modules.agents as module_agents
from modules.config_parser import ConfigParser


def get_config_from_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Training. Currently only supports image classification')
    # primary cli args
    parser.add_argument(
        '--cfg', '--config', type=str, dest="config", default="configs/classifier_cpu_config.json",
        help='config file path (default: %(default)s)')
    parser.add_argument(
        '--id', '--run_id', type=str, dest="run_id", default="train_" + datetime.now().strftime(r'%Y%m%d_%H%M%S'),
        help='unique identifier for training. Annotates train ckpts & logs. (default: %(default)s)')
    parser.add_argument(
        '-r', '--resume', type=str, dest="resume", default=None,
        help='path to resume ckpt. Overrides `resume_checkpoint` in config. (default: %(default)s)')
    parser.add_argument(
        '-o', '--override', type=str, nargs='+', dest="override", default=None,
        help='Override config params. Must match keys in json config. '
        'e.g. -o seed:1 dataset:type:DTYPE (default: %(default)s)')

    # custom cli options to modify cfg from values given in the json cfg file.
    # works in the same way as the -o/--override argument above, which takes precendent
    # to change LR {"flags": ['--lr'], "dest": "lr", "type": float, "target": "optimizer:args:lr"}
    override_options = [
        {"flags": ['--lr', '--learning_rate'],
         "dest": "learning_rate",
         "help": "Config override arg: lr param to override from config.",
         "type": float, "target": "optimizer;args;lr"},
        {"flags": ['--bs', '--train_bsize'],
         "dest": "train_bsize",
         "help": "Config override arg: train bsize to override that in config.",
         "type": int, "target": "dataloader;args;batch_size"},
        {"flags": ['--dev', '--gpu_device'],
         "dest": "gpu_device",
         "help": "Config override arg: gpu_device list i.e. None or 0, 0 1, 0 1 2.",
         "nargs": "*",
         "type": int, "target": "gpu_device"},
        {"flags": ['--mode'],
         "dest": "mode",
         "help": "Running mode. (default: %(default)s)",
         "default": "TRAIN", "choices": ["TRAIN", "TRAIN_TEST"],
         "type": str, "target": "mode"}
    ]

    config = ConfigParser.from_args(parser, override_options)
    return config


def main():
    config = get_config_from_args()
    agent = getattr(module_agents, config["trainer"]["type"])(
        config=config, logger_name="train")

    agent.train()
    if config["mode"] == "TRAIN_TEST":
        agent.test()
    agent.finalize_exit()


if __name__ == "__main__":
    main()
