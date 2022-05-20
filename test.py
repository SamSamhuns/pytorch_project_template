import argparse
from datetime import datetime

from modules.agents import classifier_agent
from modules.config_parser import ConfigParser


def get_config_from_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Test. Currently only supports image classification')
    parser.add_argument('--cfg', '--config', type=str, dest="config", default="configs/classifier_cpu_config.json",
                        help='config file path (default: %(default)s)')
    parser.add_argument('--id', '--run_id', type=str, dest="run_id", default="test_" + datetime.now().strftime(r'%Y%m%d_%H%M%S'),
                        help='unique identifier for train process. Annotates train ckpts & logs. (default: %(default)s)')
    parser.add_argument('-r', '--resume', type=str, dest="resume", required=True,
                        help='path to checkpoint ckpt for testing')

    config = ConfigParser.from_args(parser)
    return config


def main():
    config = get_config_from_args()
    agent = classifier_agent.ClassifierAgent(config, "test")

    agent.test(weight_path=config.resume)
    agent.finalize_exit()


if __name__ == "__main__":
    main()
