"""
Training script, run with: python train.py --cfg JSON_CONFIG_PATH
"""
import argparse
from datetime import datetime

from modules.agents import classifier_agent
from modules.config_parser import ConfigParser


def get_config_from_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Training. Currently only supports image classification')
    # primary cli args
    parser.add_argument('--cfg', '--config', type=str, dest="config", default="configs/classifier_cpu_config.json",
                        help='config file path (default: %(default)s)')
    parser.add_argument('--id', '--run_id', type=str, dest="run_id", default="train_" + datetime.now().strftime(r'%Y%m%d_%H%M%S'),
                        help='unique identifier for train process. Annotates train ckpts & logs. (default: %(default)s)')
    parser.add_argument('-r', '--resume', type=str, dest="resume", default=None,
                        help='path to resume ckpt. Overrides `resume_checkpoint` in config. (default: %(default)s)')

    # custom cli options to modify configuration from default values given in json file.
    # should be used to reset train params when resuming checkpoint, i.e. reducing LR
    override_options = [
        {"flags": ['--lr', '--learning_rate'],
         "dest": "learning_rate",
         "help": "lr param to override from config. (default: %(default)s)",
         "type": float, "target": "optimizer;args;lr"},
        {"flags": ['--bs', '--train_bsize'],
         "dest": "train_bsize",
         "help": "train bsize to override that in config. (default: %(default)s)",
         "type": int, "target": "dataloader;args;batch_size"},
        {"flags": ['--dev', '--gpu_device'],
         "dest": "gpu_device",
         "help": "gpu_device list i.e. None or 0, 0 1, 0 1 2. (default: %(default)s)",
         "nargs": "*",
         "type": int, "target": "gpu_device"},
        {"flags": ['--mode'],
         "dest": "mode",
         "help": "Running mode. Cannot be changed & fixed to TRAIN (default: %(default)s)",
         "default": "TRAIN", "choices": ["TRAIN"],
         "type": str, "target": "mode"}
    ]

    config = ConfigParser.from_args(parser, override_options)
    return config


def main():
    config = get_config_from_args()
    agent = classifier_agent.ClassifierAgent(config, "train")

    agent.train()
    agent.finalize_exit()


if __name__ == "__main__":
    main()
