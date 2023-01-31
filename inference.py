"""
Script for running inference with trained .pt torch models

Note: The inference script can be ported for use without a config file as well.
Use the torch checkpoint file and the necessary transforms that are used for porting
"""
import argparse
import collections
from datetime import datetime

from modules.agents import classifier_agent
from modules.config_parser import ConfigParser


def get_config_from_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Inference. Currently only supports image classification')
    # primary cli args
    parser.add_argument('--cfg', '--config', type=str, dest="config", default="configs/classifier_cpu_config.json",
                        help='config file path (default: %(default)s)')
    parser.add_argument('--id', '--run_id', type=str, dest="run_id", default="inference_" + datetime.now().strftime(r'%Y%m%d_%H%M%S'),
                        help='unique identifier for inference process when saving results. (default: %(default)s)')
    parser.add_argument('-r', '--resume', type=str, dest="resume", required=True,
                        help='path to resume ckpt for running inference.')

    # additional custom cli options
    OverrideArgs = collections.namedtuple(
        'OverrideArgs', 'flags dest help type target')
    options = [
        OverrideArgs(
            ['-s', '--source_path'],
            dest="source_path", help="path to source image file or directory for running inference. (default: %(default)s)",
            type=str, target='source_path')
    ]
    config = ConfigParser.from_args(parser, options)
    return config


def main():
    config = get_config_from_args()
    config["trainer"]["use_tensorboard"] = False
    agent = classifier_agent.ClassifierAgent(config, "inference")

    agent.inference(config["source_path"])
    agent.finalize_exit()


if __name__ == "__main__":
    main()
