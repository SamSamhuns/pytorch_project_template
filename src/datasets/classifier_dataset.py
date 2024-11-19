"""
Image Classifier Data loader
"""
import os.path as osp

import imageio.v2 as imageio
import webdataset as wds
import torchvision.utils as v_utils
from torchvision.transforms.transforms import Normalize, Compose

from src.datasets import base_dataset
from src.utils.common import identity
from src.utils.custom_statistics import get_img_dset_mean_std


def _get_webdataset_len(data_path) -> int:
    wdataset = (wds.WebDataset(data_path)
                .decode("pil")
                .to_tuple("input.jpg", "output.cls"))
    length = 0
    for _ in wdataset:
        length += 1
    return length


class ClassifierDataset:
    def __init__(self,
                 train_transform=None,
                 val_transform=None,
                 test_transform=None,
                 data_mode="imgs",
                 root='data',
                 train_path='train',
                 val_path='val',
                 test_path='test',
                 **kwargs):
        """
        train_transform: torchvision.transforms for train data
        val_transform: torchvision.transforms for validation data
        test_transform: torchvision.transforms for test data
        data_mode: Mode for getting data
        root: folder containing train & test data dirs
        train_path: train dir under root
        val_path: val dir under root
        test_path: test dir under root

        data_root
                |--train_path
                |--val_path
                |--test_path
        """
        self.val_set = None
        self.test_set = None
        if data_mode == "imgs":
            train_root = osp.join(root, train_path)
            self.train_set = base_dataset.ImageFolderDataset(
                train_root, transforms=train_transform)
            if val_path is not None:
                val_root = osp.join(root, val_path)
                self.val_set = base_dataset.ImageFolderDataset(
                    val_root, transforms=val_transform)
            if test_path is not None:
                test_root = osp.join(root, test_path)
                self.test_set = base_dataset.ImageFolderDataset(
                    test_root, transforms=test_transform)
        elif data_mode == "webdataset":
            train_root = osp.join(root, train_path)
            train_len = _get_webdataset_len(train_root)
            self.train_set = (wds.WebDataset(train_root)
                              .shuffle(100)
                              .decode("pil")
                              .to_tuple("input.jpg", "output.cls")
                              .map_tuple(train_transform, identity)
                              .with_length(train_len))
            if val_path is not None:
                val_root = osp.join(root, val_path)
                val_len = _get_webdataset_len(val_root)
                self.val_set = (wds.WebDataset(val_root)
                                .shuffle(100)
                                .decode("pil")
                                .to_tuple("input.jpg", "output.cls")
                                .map_tuple(val_transform, identity)
                                .with_length(val_len))
            if test_path is not None:
                test_root = osp.join(root, test_path)
                test_len = _get_webdataset_len(test_root)
                self.test_set = (wds.WebDataset(test_root)
                                 .shuffle(100)
                                 .decode("pil")
                                 .to_tuple("input.jpg", "output.cls")
                                 .map_tuple(test_transform, identity)
                                 .with_length(test_len))
        elif data_mode == "numpy":
            raise NotImplementedError(
                "This mode is not implemented YET")
        else:
            raise NotImplementedError(
                f"{data_mode} data_mode is not supported. Available modes are: imgs, webdataset")

        # if Normalize transforms absent in train_tfs, calc from the train data & assign to train,test,val
        if not any(tfs for tfs in self.train_set.transforms.transforms if isinstance(tfs, Normalize)):
            mean, std = get_img_dset_mean_std(self.train_set, method="online")
            norm = Normalize(mean.tolist(), std.tolist())

            self.train_set.transforms = Compose([*self.train_set.transforms.transforms, norm])
            if self.val_set is not None:
                self.val_set.transforms = Compose([*self.val_set.transforms.transforms, norm])
            if self.test_set is not None:
                self.test_set.transforms = Compose([*self.test_set.transforms.transforms, norm])

    def plot_samples_per_epoch(self, batch, epoch, out_dir):
        """
        Plotting the batch images
        :param batch: Tensor of shape (B,C,H,W)
        :param epoch: the number of current epoch
        :param out_dir: output save directory
        :return: img_epoch: which will contain the image of this epoch
        """
        img_epoch = f"{out_dir}samples_epoch_{epoch:d}.png"
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
            img_epoch = f'{out_dir}samples_epoch_{epoch:d}.png'
            try:
                gen_image_plots.append(imageio.imread(img_epoch))
            except OSError as e:
                print(e)

        imageio.mimsave(
            out_dir + f'animation_epochs_{epochs:d}.gif', gen_image_plots, fps=2)

    def finalize(self):
        pass
