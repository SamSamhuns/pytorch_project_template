"""
Mnist Data loader
"""
import imageio
import torchvision.utils as v_utils
from torchvision import datasets, transforms


class MnistDataset:
    def __init__(self, data_mode):
        """
        :param config:
        """
        if data_mode == "download":
            self.train_dataset = datasets.MNIST('data',
                                                train=True,
                                                download=True,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        (0.1307,), (0.3081,))
                                                ]))
            self.test_dataset = datasets.MNIST('data',
                                               train=False,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(
                                                       (0.1307,), (0.3081,))
                                               ]))
        elif data_mode == "imgs":
            raise NotImplementedError("This mode is not implemented YET")

        elif data_mode == "numpy":
            raise NotImplementedError("This mode is not implemented YET")

        else:
            raise Exception(
                "Please specify in the json a specified mode in data_mode")

    def plot_samples_per_epoch(self, batch, epoch, out_dir):
        """
        Plotting the batch images
        :param batch: Tensor of shape (B,C,H,W)
        :param epoch: the number of current epoch
        :param out_dir: output save directory
        :return: img_epoch: which will contain the image of this epoch
        """
        img_epoch = '{}samples_epoch_{:d}.png'.format(out_dir, epoch)
        v_utils.save_image(batch,
                           img_epoch,
                           nrow=4,
                           padding=2,
                           normalize=True)
        return imageio.imread(img_epoch)

    def make_gif(self, epochs, out_dir):
        """
        Make a gif from a multiple images of epochs
        :param epochs: num_epochs till now
        :param out_dir: output save directory
        :return:
        """
        gen_image_plots = []
        for epoch in range(epochs + 1):
            img_epoch = '{}samples_epoch_{:d}.png'.format(out_dir, epoch)
            try:
                gen_image_plots.append(imageio.imread(img_epoch))
            except OSError as e:
                print(e)

        imageio.mimsave(
            out_dir + 'animation_epochs_{:d}.gif'.format(epochs), gen_image_plots, fps=2)

    def finalize(self):
        pass
