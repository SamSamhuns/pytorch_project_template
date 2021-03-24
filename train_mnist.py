from modules.agents import mnist_agent
from configs import mnist_config


def main():

    agent = mnist_agent.MnistAgent(mnist_config.MNIST_CONFIG)
    agent.train()
    # agent.finalize_exit()


if __name__ == "__main__":
    main()
