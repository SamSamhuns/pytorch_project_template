"""
Base Dataset Class
"""
import os
import os.path as osp
from typing import List, Dict

from PIL import Image
from torch.utils import data


IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.webp'}


def is_file_ext_valid(filepath: str, extensions: List[str]):
    """
    check if a filepath has allowed extensions
    """
    filepath_lower = filepath.lower()
    return any(filepath_lower.endswith(ext) for ext in extensions)


def _find_classes(root_dir: str):
    """returns tuple class names & class to idx dicts from root_dir
    root_dir struct:
        root
            |_ class_x
                      |_ x1.ext
                      |_ x2.ext
            |_ class_y
                      |_ y1.ext
                      |_ y2.ext
    """
    # list of classes or subfolders under root_dir
    classes = [d for d in sorted(os.listdir(root_dir))
               if osp.isdir(osp.join(root_dir, d))]

    # class_name to index dict, order is alphabetical sorting of class names
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def _make_dataset(directory: str,
                  class_to_idx: Dict[str, int],
                  extensions: List[str]):
    """
    returns a list of img_path, class index tuples
    """
    images = []
    directory = osp.expanduser(directory)
    for target in sorted(os.listdir(directory)):
        d = osp.join(directory, target)
        if not osp.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_file_ext_valid(fname, extensions):
                    path = osp.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images


def get_pil_img(path):
    """
    open path as file to avoid ResourceWarning
    """
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as fread:
        img = Image.open(fread)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return get_pil_img(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return get_pil_img(path)


def write_class_mapping_to_file(mapping, fpath) -> None:
    """
    mapping must be one level deep dict of class_names to index
    """
    with open(fpath, 'w', encoding="utf-8") as fw:
        for class_name, class_idx in mapping.items():
            fw.write(''.join([class_name, ' ', str(class_idx), '\n']))


class BaseDataset(data.Dataset):
    """
    A generic data loader where the samples are arranged as follows:
    Note: ImageNet style class_dir->subdirs->subdirs->images... is also supported
        root
            |_ class_x
                      |_ x1.ext
                      |_ x2.ext
                      |_ x3.ext
            |_ class_y
                      |_ y1.ext
                      |_ y2.ext
                      |_ y3.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transforms (callable, optional): Function/transforms that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transforms (callable, optional): Function/transforms that takes
            in the target and transforms it.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self,
                 root,
                 loader,
                 extensions,
                 transforms=None,
                 target_transforms=None):
        classes, class_to_idx = _find_classes(root)
        data = _make_dataset(root, class_to_idx, extensions)
        if len(data) == 0:
            raise (RuntimeError(
                f"Found 0 files in subfolders of: {root}"
                f"\nSupported extensions are: {','.join(extensions)}"))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.data = data

        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target)
            where target is class_index of the target class.

            None when there is an error loading a datum
        """
        if index < 0 or index >= len(self.data):
            raise IndexError(f"Index {index} is out of range for dataset of size {len(self.data)}.")
        try:
            path, target = self.data[index]
            sample = self.loader(path)
            if self.transforms is not None:
                sample = self.transforms(sample)
            if self.target_transforms is not None:
                target = self.target_transforms(target)

            return sample, target
        except Exception as excep:
            print(excep)
            return None

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += f'    Root Location: {self.root}\n'
        fmt_str += f'    Number of datapoints: {self.__len__()}\n'
        fmt_str += f'    Number of classes: {len(self.classes)}\n'
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class ImageFolderDataset(BaseDataset):
    """
    An Image data loader where the images are arranged as follows:
    Note: ImageNet style class_dir->subdirs->subdirs->images... is also supported
        root
            |_ class_x
                      |_ x1.ext
                      |_ x2.ext
                      |_ x3.ext
            |_ class_y
                      |_ y1.ext
                      |_ y2.ext
                      |_ y3.ext
    Args:
        root (string): Root directory path.
        transforms (callable, optional):
            Function/transforms that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transforms (callable, optional):
            A function/transforms that takes in the target and transforms it.
        loader (callable, optional):
            A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root,
                 file_extensions=IMG_EXTENSIONS,
                 transforms=None,
                 target_transforms=None,
                 loader=default_loader):
        super().__init__(
            root,
            loader,
            file_extensions,
            transforms=transforms,
            target_transforms=target_transforms)
        self.imgs = self.data


if __name__ == "__main__":
    from torchvision import transforms

    train_data = ImageFolderDataset(
        "data/birds_dataset/valid",
        IMG_EXTENSIONS,
        transforms=transforms.Compose([
            transforms.CenterCrop(299),
            transforms.ColorJitter(brightness=1.5,
                                   contrast=0.8,
                                   saturation=0.8,
                                   hue=0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]))
    print("Printing ImageFolderDataset data class", train_data)
    print("Printing len of data class (Number of images in data)", len(train_data))
    print("Printing the shape of the first datum and its label from the data",
          train_data[0][0].shape, train_data[0][1])

    print()
