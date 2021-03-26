from modules.agents import classifier_agent
from configs import mnist_config


def main():

    c_agent = classifier_agent.ClassifierAgent(mnist_config.MNIST_CONFIG)
    c_agent.train()
    # mnist_agent.finalize_exit()


if __name__ == "__main__":
    main()
