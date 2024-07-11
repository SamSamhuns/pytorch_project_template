import os
import tempfile
import torch
from PIL import Image
import pytest

from tests.conftest import NUM_CLS, NUM_IMGS_P_CLS, PYTEST_TEMP_ROOT
from src.datasets import ImageFolderDataset
from src.datasets.base_dataset import (
    is_file_ext_valid, _find_classes, _make_dataset, get_pil_img, write_class_mapping_to_file)



def test_is_file_ext_valid():
    """Test with a valid and an invalid extension"""
    assert is_file_ext_valid("image.jpg", ['.jpg', '.jpeg', '.png'])
    assert not is_file_ext_valid("image.txt", ['.jpg', '.jpeg', '.png'])


def test_find_classes():
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.makedirs(os.path.join(tmpdirname, 'class_x'))
        os.makedirs(os.path.join(tmpdirname, 'class_y'))
        classes, class_to_idx = _find_classes(tmpdirname)

        assert set(classes) == {'class_x', 'class_y'}
        assert class_to_idx == {'class_x': 0, 'class_y': 1}


def test_make_dataset():
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.makedirs(os.path.join(tmpdirname, 'class_x'))
        os.makedirs(os.path.join(tmpdirname, 'class_y'))
        # Creating dummy image files
        with open(os.path.join(tmpdirname, 'class_x', 'x1.jpg'), 'w', encoding="utf-8") as f:
            f.write("Fake image content")
        with open(os.path.join(tmpdirname, 'class_y', 'y1.png'), 'w', encoding="utf-8") as f:
            f.write("Fake image content")

        class_to_idx = {'class_x': 0, 'class_y': 1}
        dataset = _make_dataset(tmpdirname, class_to_idx, ['.jpg', '.png'])

        assert len(dataset) == 2
        assert (os.path.join(tmpdirname, 'class_x', 'x1.jpg'), 0) in dataset
        assert (os.path.join(tmpdirname, 'class_y', 'y1.png'), 1) in dataset


def test_get_pil_img():
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmpfile:
        # Create a small image file
        img = Image.new('RGB', (10, 10))
        img.save(tmpfile.name)

        # Open it using our function
        opened_img = get_pil_img(tmpfile.name)
        assert opened_img.mode == 'RGB'


def test_default_loader():
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmpfile:
        # Create a small image file
        img = Image.new('RGB', (10, 10))
        img.save(tmpfile.name)


def test_initialization(image_dataset):
    """Check if the dataset is initialized correctly"""
    assert len(image_dataset) == NUM_CLS * NUM_IMGS_P_CLS
    assert len(image_dataset.classes) == NUM_CLS


def test_getitem(image_dataset):
    """Test __getitem__ and transformations"""
    img, label = image_dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 100, 100)  # Check shape after ToTensor
    assert isinstance(label, int)  # Check if label is integer


def test_nonexistent_directory():
    with pytest.raises(FileNotFoundError):
        ImageFolderDataset('/path/does/not/exist')


def test_empty_directory():
    """Test initialization with an empty directory"""
    empty_dir = os.path.join(PYTEST_TEMP_ROOT, "empty")
    os.makedirs(empty_dir, exist_ok=True)
    with pytest.raises(RuntimeError):
        ImageFolderDataset(str(empty_dir))


def test_repr(image_dataset):
    """Check the string representation of the dataset for expected info"""
    repr_string = repr(image_dataset)
    assert 'ImageFolderDataset' in repr_string
    assert f'Number of datapoints: {NUM_CLS * NUM_IMGS_P_CLS}' in repr_string
    assert f'Number of classes: {NUM_CLS}' in repr_string


def test_write_class_mapping_to_file(image_dataset):
    """Test writing class mapping to file"""
    mapping_path = os.path.join(PYTEST_TEMP_ROOT, "class_mapping.txt")
    write_class_mapping_to_file(image_dataset.class_to_idx, mapping_path)
    assert os.path.exists(mapping_path)
    with open(mapping_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == NUM_CLS
