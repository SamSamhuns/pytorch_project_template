"""
Image Classifier Data loader
"""
import os.path as osp

import imageio
import webdataset as wds
import torchvision.utils as v_utils

from modules.datasets import base_dataset


class ClassifierDataset:
    def __init__(self,
                 train_transform=None,
                 val_transform=None,
                 test_transform=None,
                 data_mode="imgs",
                 data_root='data',
                 train_dir='train',
                 val_dir='val',
                 test_dir='test',
                 **kwargs):
        """
        train_transform: torchvision.transforms for train data
        val_transform: torchvision.transforms for validation data
        test_transform: torchvision.transforms for test data
        data_mode: Mode for getting data
            data_root: folder containing train & test data dirs
            train_dir: train dir under data_root
            val_dir: val dir under data_root
            test_dir: test dir under data_root

        data_root
                |--train_dir
                |--val_dir
                |--test_dir
        """
        if data_mode == "imgs":
            train_root = osp.join(data_root, train_dir)
            self.train_dataset = base_dataset.ImageFolderDataset(train_root,
                                                                 transform=train_transform)
            if val_dir is not None:
                val_root = osp.join(data_root, val_dir)
                self.val_dataset = base_dataset.ImageFolderDataset(val_root,
                                                                   transform=val_transform)
            if test_dir is not None:
                test_root = osp.join(data_root, test_dir)
                self.test_dataset = base_dataset.ImageFolderDataset(test_root,
                                                                    transform=test_transform)
        elif data_mode == "webdataset":
            train_root = osp.join(data_root, train_dir)
            self.train_dataset = wds.WebDataset(train_root).shuffle(1000).decode("torchrgb").to_tuple("input.jpg", "output.cls")
            if val_dir is not None:
                val_root = osp.join(data_root, val_dir)
                self.val_dataset = wds.WebDataset(val_root).shuffle(1000).decode("torchrgb").to_tuple("input.jpg", "output.cls")
            if test_dir is not None:
                test_root = osp.join(data_root, test_dir)
                self.test_dataset = wds.WebDataset(test_root).shuffle(1000).decode("torchrgb").to_tuple("input.jpg", "output.cls")
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
