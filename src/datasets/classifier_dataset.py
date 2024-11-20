"""
Image Classifier Data loader
"""
import os.path as osp

import imageio.v2 as imageio
import webdataset as wds
import torchvision.utils as v_utils
from torchvision.transforms.transforms import Normalize, Compose

from src.datasets import base_dataset
from src.utils.common import identity, add_tfms
from src.utils.custom_statistics import get_img_dset_mean_std


def _get_webdataset_len(data_path) -> int:
    wdataset = (wds.WebDataset(data_path, shardshuffle=False)
                .decode("pil")
                .to_tuple("input.jpg", "output.cls"))
    length = 0
    for _ in wdataset:
        length += 1
    return length


def _get_webdataset(dset_path: str, tfms: Compose, dset_len: int, decode: str = "pil"):
    """
    Get a webdataset
    """
    return (wds.WebDataset(dset_path, shardshuffle=True)
            .shuffle(1000)
            .decode(decode)
            .to_tuple("input.jpg", "output.cls")
            .map_tuple(tfms, identity)
            .with_length(dset_len))


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
            # add normalization transforms if not present
            if not any(isinstance(t, Normalize) for t in train_transform.transforms):
                _temp_train_set = base_dataset.ImageFolderDataset(
                    train_root, transforms=train_transform)
                print("Normalization tfms not found. Calculating mean and std for the dataset.")
                mean, std = get_img_dset_mean_std(_temp_train_set, method="online")
                normalize = Normalize(mean.tolist(), std.tolist())
                train_transform = add_tfms(train_transform, normalize)
                val_transform = add_tfms(val_transform, normalize)
                test_transform = add_tfms(test_transform, normalize)

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
            # add normalization transforms if not present
            if not any(isinstance(t, Normalize) for t in train_transform.transforms):
                _temp_train_set = _get_webdataset(train_root, train_transform, train_len)
                print("Normalization tfms not found. Calculating mean and std for the dataset.")
                mean, std = get_img_dset_mean_std(_temp_train_set, method="online")
                normalize = Normalize(mean.tolist(), std.tolist())
                train_transform = add_tfms(train_transform, normalize)
                val_transform = add_tfms(val_transform, normalize)
                test_transform = add_tfms(test_transform, normalize)

            self.train_set = _get_webdataset(train_root, train_transform, train_len)
            if val_path is not None:
                val_root = osp.join(root, val_path)
                val_len = _get_webdataset_len(val_root)
                self.val_set = _get_webdataset(val_root, val_transform, val_len)
            if test_path is not None:
                test_root = osp.join(root, test_path)
                test_len = _get_webdataset_len(test_root)
                self.test_set = _get_webdataset(test_root, test_transform, test_len)
        elif data_mode == "numpy":
            raise NotImplementedError(
                "This mode is not implemented YET")
        else:
            raise NotImplementedError(
                f"{data_mode} data_mode is not supported. Available modes are: imgs, webdataset")
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

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
