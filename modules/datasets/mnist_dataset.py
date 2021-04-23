"""
Mnist Data loader
"""
import imageio
import torchvision.utils as v_utils
from torchvision import datasets


class MnistDataset:
    def __init__(self,
                 train_transform=None,
                 test_transform=None,
                 data_root='data',
                 data_mode="download",
                 **kwargs):
        """
        train_transform: torchvision.transforms for train data
        test_transform: torchvision.transforms for test data
        data_root: folder containing train & test data dirs
        data_mode: Mode for getting data
        """
        if data_mode == "download":
            self.train_dataset = datasets.MNIST(data_root,
                                                train=True,
                                                download=True,
                                                transform=train_transform)
            self.test_dataset = datasets.MNIST(data_root,
                                               train=False,
                                               transform=test_transform)
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
